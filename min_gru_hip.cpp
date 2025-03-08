#include "min_gru.h"
#include "parallel_scan.h"
#include <hip/hip_runtime.h>

// HIP error checking macro
#define CHECK_HIP_ERROR(call) { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error in %s at line %d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// External functions defined in parallel_scan_hip.cpp
extern "C" void parallel_scan_hip(int seq_length, int batch_size, int hidden_size,
                            const float* a, const float* b, const float* h0,
                            float* h_out);

// External kernel declarations (defined in parallel_scan_hip.cpp)
extern __global__ void compose_scan_ops_kernel(const float* a1, const float* b1,
                                             const float* a2, const float* b2,
                                             float* a_out, float* b_out,
                                             int hidden_size);

extern __global__ void apply_scan_op_kernel(const float* a, const float* b,
                                          const float* h_in, float* h_out,
                                          int hidden_size);

// HIP kernel for MinGRU forward pass
__global__ void min_gru_forward_kernel(const LinearLayer* d_linear_z, const LinearLayer* d_linear_h,
                                      const float* x_t, const float* h_prev, float* h_t,
                                      int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < hidden_size) {
        // Temporary variables for this thread's computation
        float z_t = 0.0f;
        float h_tilde = 0.0f;
        
        // Compute update gate: z_t = sigmoid(Linear_z(x_t))
        // We assume x_t is already in device memory
        // Each thread computes one element of the output vector
        z_t = d_linear_z->bias[idx];
        for (int i = 0; i < d_linear_z->input_size; i++) {
            z_t += d_linear_z->weights[idx * d_linear_z->input_size + i] * x_t[i];
        }
        z_t = 1.0f / (1.0f + expf(-z_t)); // sigmoid
        
        // Compute candidate hidden state: h_tilde = Linear_h(x_t)
        h_tilde = d_linear_h->bias[idx];
        for (int i = 0; i < d_linear_h->input_size; i++) {
            h_tilde += d_linear_h->weights[idx * d_linear_h->input_size + i] * x_t[i];
        }
        
        // Compute h_t = (1 - z_t) * h_prev + z_t * h_tilde
        h_t[idx] = (1.0f - z_t) * h_prev[idx] + z_t * h_tilde;
    }
}

// HIP kernel for preparing parallel scan parameters
__global__ void min_gru_extract_scan_params_kernel(LinearLayer d_linear_z, 
                                                  LinearLayer d_linear_h,
                                                  const float* x, float* a, float* b,
                                                  int seq_length, int hidden_size, int input_size) {
    // Each thread handles one element of the hidden state for one time step
    int t = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds checking
    if (t >= seq_length || h >= hidden_size) {
        return;
    }
    
    // Additional pointer checks for kernel inputs
    if (!x || !a || !b) {
        return;
    }
    
    if (!d_linear_z.weights || !d_linear_z.bias || !d_linear_h.weights || !d_linear_h.bias) {
        return;
    }
    
    // Get input for current time step
    const float* x_t = x + t * input_size;
    
    // Compute update gate: z_t = sigmoid(Linear_z(x_t))
    float z_t = d_linear_z.bias[h];
    for (int i = 0; i < input_size; i++) {
        z_t += d_linear_z.weights[h * input_size + i] * x_t[i];
    }
    z_t = 1.0f / (1.0f + expf(-z_t)); // sigmoid
    
    // Compute candidate hidden state: h_tilde = Linear_h(x_t)
    float h_tilde = d_linear_h.bias[h];
    for (int i = 0; i < input_size; i++) {
        h_tilde += d_linear_h.weights[h * input_size + i] * x_t[i];
    }
    
    // Compute a_t = 1 - z_t and b_t = z_t * h_tilde
    a[t * hidden_size + h] = 1.0f - z_t;
    b[t * hidden_size + h] = z_t * h_tilde;
}

// HIP implementation of the parallel scan algorithm for minGRU - just a wrapper around the general one
extern "C" void min_gru_parallel_scan_hip(int seq_length, int batch_size, int hidden_size,
                              const float* a, const float* b, const float* h0,
                              float* h_out) {
    // This is now a wrapper that calls the shared implementation
    parallel_scan_hip(seq_length, batch_size, hidden_size, a, b, h0, h_out);
}

// Helper function to transfer MinGRU cell to device
void min_gru_to_device(const MinGRUCell* cell, LinearLayer* d_linear_z, LinearLayer* d_linear_h) {
    fprintf(stderr, "DEBUG: In min_gru_to_device - input_size=%d, hidden_size=%d\n", 
            cell->input_size, cell->hidden_size);
    
    // Initialize all fields to zero/NULL first
    d_linear_z->weights = NULL;
    d_linear_z->bias = NULL;
    d_linear_z->input_size = 0;
    d_linear_z->output_size = 0;
    
    d_linear_h->weights = NULL;
    d_linear_h->bias = NULL; 
    d_linear_h->input_size = 0;
    d_linear_h->output_size = 0;
    
    // Allocate and copy linear_z
    fprintf(stderr, "DEBUG: Allocating d_linear_z weights (%zu bytes)\n", 
            cell->hidden_size * cell->input_size * sizeof(float));
    
    hipError_t err = hipMalloc((void**)&d_linear_z->weights, 
                         cell->hidden_size * cell->input_size * sizeof(float));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc for d_linear_z->weights failed: %s\n", hipGetErrorString(err));
        return;
    }
    
    fprintf(stderr, "DEBUG: Allocating d_linear_z bias (%zu bytes)\n", 
            cell->hidden_size * sizeof(float));
    
    err = hipMalloc((void**)&d_linear_z->bias, cell->hidden_size * sizeof(float));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc for d_linear_z->bias failed: %s\n", hipGetErrorString(err));
        hipFree(d_linear_z->weights);
        d_linear_z->weights = NULL;
        return;
    }
    
    fprintf(stderr, "DEBUG: Copying d_linear_z weights to device\n");
    err = hipMemcpy(d_linear_z->weights, cell->linear_z.weights, 
                   cell->hidden_size * cell->input_size * sizeof(float), 
                   hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy for d_linear_z weights failed: %s\n", hipGetErrorString(err));
        hipFree(d_linear_z->weights);
        hipFree(d_linear_z->bias);
        d_linear_z->weights = NULL;
        d_linear_z->bias = NULL;
        return;
    }
    
    fprintf(stderr, "DEBUG: Copying d_linear_z bias to device\n");
    err = hipMemcpy(d_linear_z->bias, cell->linear_z.bias, 
                   cell->hidden_size * sizeof(float), 
                   hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy for d_linear_z bias failed: %s\n", hipGetErrorString(err));
        hipFree(d_linear_z->weights);
        hipFree(d_linear_z->bias);
        d_linear_z->weights = NULL;
        d_linear_z->bias = NULL;
        return;
    }
    
    // Set the dimensions in the device structs
    d_linear_z->input_size = cell->input_size;
    d_linear_z->output_size = cell->hidden_size;
    fprintf(stderr, "DEBUG: d_linear_z setup complete\n");
    
    // Allocate and copy linear_h
    fprintf(stderr, "DEBUG: Allocating d_linear_h weights (%zu bytes)\n", 
            cell->hidden_size * cell->input_size * sizeof(float));
    
    err = hipMalloc((void**)&d_linear_h->weights, 
                    cell->hidden_size * cell->input_size * sizeof(float));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc for d_linear_h->weights failed: %s\n", hipGetErrorString(err));
        hipFree(d_linear_z->weights);
        hipFree(d_linear_z->bias);
        return;
    }
    
    fprintf(stderr, "DEBUG: Allocating d_linear_h bias (%zu bytes)\n", 
            cell->hidden_size * sizeof(float));
    
    err = hipMalloc((void**)&d_linear_h->bias, cell->hidden_size * sizeof(float));
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc for d_linear_h->bias failed: %s\n", hipGetErrorString(err));
        hipFree(d_linear_z->weights);
        hipFree(d_linear_z->bias);
        hipFree(d_linear_h->weights);
        return;
    }
    
    fprintf(stderr, "DEBUG: Copying d_linear_h weights to device\n");
    err = hipMemcpy(d_linear_h->weights, cell->linear_h.weights, 
                   cell->hidden_size * cell->input_size * sizeof(float), 
                   hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy for d_linear_h weights failed: %s\n", hipGetErrorString(err));
        hipFree(d_linear_z->weights);
        hipFree(d_linear_z->bias);
        hipFree(d_linear_h->weights);
        hipFree(d_linear_h->bias);
        return;
    }
    
    fprintf(stderr, "DEBUG: Copying d_linear_h bias to device\n");
    err = hipMemcpy(d_linear_h->bias, cell->linear_h.bias, 
                   cell->hidden_size * sizeof(float), 
                   hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy for d_linear_h bias failed: %s\n", hipGetErrorString(err));
        hipFree(d_linear_z->weights);
        hipFree(d_linear_z->bias);
        hipFree(d_linear_h->weights);
        hipFree(d_linear_h->bias);
        return;
    }
    
    // Set the dimensions in the device struct
    d_linear_h->input_size = cell->input_size;
    d_linear_h->output_size = cell->hidden_size;
    
    fprintf(stderr, "DEBUG: d_linear_h setup complete\n");
    fprintf(stderr, "DEBUG: min_gru_to_device completed successfully\n");
}

// Helper function to free device memory for a MinGRU cell
void min_gru_free_device(LinearLayer* d_linear_z, LinearLayer* d_linear_h) {
    fprintf(stderr, "DEBUG: Freeing device memory for MinGRU cell\n");
    
    // Free linear_z device memory with NULL checks
    if (d_linear_z) {
        if (d_linear_z->weights) {
            fprintf(stderr, "DEBUG: Freeing d_linear_z->weights\n");
            hipError_t err = hipFree(d_linear_z->weights);
            if (err != hipSuccess) {
                fprintf(stderr, "WARNING: hipFree for d_linear_z->weights failed: %s\n", hipGetErrorString(err));
            }
            d_linear_z->weights = NULL;
        }
        
        if (d_linear_z->bias) {
            fprintf(stderr, "DEBUG: Freeing d_linear_z->bias\n");
            hipError_t err = hipFree(d_linear_z->bias);
            if (err != hipSuccess) {
                fprintf(stderr, "WARNING: hipFree for d_linear_z->bias failed: %s\n", hipGetErrorString(err));
            }
            d_linear_z->bias = NULL;
        }
    }
    
    // Free linear_h device memory with NULL checks
    if (d_linear_h) {
        if (d_linear_h->weights) {
            fprintf(stderr, "DEBUG: Freeing d_linear_h->weights\n");
            hipError_t err = hipFree(d_linear_h->weights);
            if (err != hipSuccess) {
                fprintf(stderr, "WARNING: hipFree for d_linear_h->weights failed: %s\n", hipGetErrorString(err));
            }
            d_linear_h->weights = NULL;
        }
        
        if (d_linear_h->bias) {
            fprintf(stderr, "DEBUG: Freeing d_linear_h->bias\n");
            hipError_t err = hipFree(d_linear_h->bias);
            if (err != hipSuccess) {
                fprintf(stderr, "WARNING: hipFree for d_linear_h->bias failed: %s\n", hipGetErrorString(err));
            }
            d_linear_h->bias = NULL;
        }
    }
    
    fprintf(stderr, "DEBUG: MinGRU device memory freed\n");
}

// Process a full sequence with MinGRU using HIP
extern "C" void min_gru_process_sequence_hip(const MinGRUCell* cell, const float* x, const float* h0,
                                int seq_length, float* h_out) {
    fprintf(stderr, "DEBUG: Starting min_gru_process_sequence_hip\n");
    
    // Setup kernel launch parameters (moved up)
    int threadsPerBlock = 256;
    int blocksPerGrid = (cell->hidden_size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 extractBlockDim(threadsPerBlock);
    dim3 extractGridDim(blocksPerGrid, seq_length);
    
    // For debugging with smaller grid
    int simpleBlockSize = 32; // Use a smaller block size
    dim3 debugBlockDim(simpleBlockSize);
    dim3 debugGridDim((cell->hidden_size + simpleBlockSize - 1) / simpleBlockSize, 1); // Process just 1 timestep first
    
    // Transfer cell to device
    fprintf(stderr, "DEBUG: Transferring cell to device\n");
    LinearLayer d_linear_z, d_linear_h;
    min_gru_to_device(cell, &d_linear_z, &d_linear_h);
    fprintf(stderr, "DEBUG: Cell transferred to device\n");
    
    // Allocate device memory for input/output
    fprintf(stderr, "DEBUG: Allocating device memory\n");
    float *d_x = NULL, *d_h0 = NULL, *d_h_out = NULL;
    float *d_a = NULL, *d_b = NULL;
    
    size_t input_size_bytes = seq_length * cell->input_size * sizeof(float);
    size_t hidden_size_bytes = cell->hidden_size * sizeof(float);
    size_t seq_hidden_size = seq_length * cell->hidden_size * sizeof(float);
    
    hipError_t err;
    
    fprintf(stderr, "DEBUG: Allocating d_x (%zu bytes)\n", input_size_bytes);
    err = hipMalloc((void**)&d_x, input_size_bytes);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc for d_x failed: %s\n", hipGetErrorString(err));
        goto cleanup; // Now safe as all variables are initialized
    }
    
    fprintf(stderr, "DEBUG: Allocating d_h0 (%zu bytes)\n", hidden_size_bytes);
    err = hipMalloc((void**)&d_h0, hidden_size_bytes);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc for d_h0 failed: %s\n", hipGetErrorString(err));
        goto cleanup;
    }
    
    fprintf(stderr, "DEBUG: Allocating d_h_out (%zu bytes)\n", seq_hidden_size);
    err = hipMalloc((void**)&d_h_out, seq_hidden_size);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc for d_h_out failed: %s\n", hipGetErrorString(err));
        goto cleanup;
    }
    
    fprintf(stderr, "DEBUG: Allocating d_a (%zu bytes)\n", seq_hidden_size);
    err = hipMalloc((void**)&d_a, seq_hidden_size);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc for d_a failed: %s\n", hipGetErrorString(err));
        goto cleanup;
    }
    
    fprintf(stderr, "DEBUG: Allocating d_b (%zu bytes)\n", seq_hidden_size);
    err = hipMalloc((void**)&d_b, seq_hidden_size);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc for d_b failed: %s\n", hipGetErrorString(err));
        goto cleanup;
    }
    fprintf(stderr, "DEBUG: Device memory allocated\n");
    
    // Copy input data to device
    fprintf(stderr, "DEBUG: Copying input data to device\n");
    err = hipMemcpy(d_x, x, input_size_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy for d_x failed: %s\n", hipGetErrorString(err));
        goto cleanup;
    }
    
    err = hipMemcpy(d_h0, h0, hidden_size_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy for d_h0 failed: %s\n", hipGetErrorString(err));
        goto cleanup;
    }
    fprintf(stderr, "DEBUG: Input data copied to device\n");
    
    // Extract scan parameters
    fprintf(stderr, "DEBUG: Extracting scan parameters\n");
    
    // Use smaller, safer grid dimensions
    fprintf(stderr, "DEBUG: Grid dimensions: blocks=%d, threads=%d, seq_length=%d\n", 
            blocksPerGrid, threadsPerBlock, seq_length);
    fprintf(stderr, "DEBUG: Hidden size=%d, Input size=%d\n", cell->hidden_size, cell->input_size);
    
    // Print the LinearLayer structure information
    fprintf(stderr, "DEBUG: d_linear_z weights=%p, bias=%p\n", 
            (void*)d_linear_z.weights, (void*)d_linear_z.bias);
    fprintf(stderr, "DEBUG: d_linear_h weights=%p, bias=%p\n", 
            (void*)d_linear_h.weights, (void*)d_linear_h.bias);
    fprintf(stderr, "DEBUG: Other pointers: d_x=%p, d_a=%p, d_b=%p\n", 
            (void*)d_x, (void*)d_a, (void*)d_b);
    
    fprintf(stderr, "DEBUG: Using simplified grid/block for debugging: grid=(%d,%d), block=(%d)\n", 
            debugGridDim.x, debugGridDim.y, debugBlockDim.x);
    
    // Launch kernel with simplified dimensions to test first timestep
    fprintf(stderr, "DEBUG: Launching min_gru_extract_scan_params_kernel for first timestep only\n");
    hipLaunchKernelGGL(min_gru_extract_scan_params_kernel, debugGridDim, debugBlockDim, 0, 0,
        d_linear_z, d_linear_h, d_x, d_a, d_b, 
        1, // Process only 1 timestep for debugging
        cell->hidden_size, cell->input_size
    );
    
    // Check for kernel launch errors
    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: Kernel launch failed: %s\n", hipGetErrorString(err));
        goto cleanup;
    }
    
    // Wait for kernel to finish
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipDeviceSynchronize failed: %s\n", hipGetErrorString(err));
        goto cleanup;
    }
    fprintf(stderr, "DEBUG: First timestep scan parameters extracted successfully\n");
    
    // Now that first timestep works, process all time steps
    if (seq_length > 1) {
        fprintf(stderr, "DEBUG: Processing all %d timesteps\n", seq_length);
        hipLaunchKernelGGL(min_gru_extract_scan_params_kernel, extractGridDim, extractBlockDim, 0, 0,
            d_linear_z, d_linear_h, d_x, d_a, d_b, 
            seq_length, cell->hidden_size, cell->input_size
        );
        
        err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "ERROR: Full kernel launch failed: %s\n", hipGetErrorString(err));
            goto cleanup;
        }
        
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            fprintf(stderr, "ERROR: hipDeviceSynchronize failed for full kernel: %s\n", hipGetErrorString(err));
            goto cleanup;
        }
        fprintf(stderr, "DEBUG: All scan parameters extracted successfully\n");
    }

    // Run parallel scan
    fprintf(stderr, "DEBUG: Running min_gru_parallel_scan_hip\n");
    min_gru_parallel_scan_hip(seq_length, 1, cell->hidden_size, d_a, d_b, d_h0, d_h_out);
    fprintf(stderr, "DEBUG: Parallel scan completed\n");

    // Copy results back to host
    fprintf(stderr, "DEBUG: Copying results back to host\n");
    err = hipMemcpy(h_out, d_h_out, seq_length * cell->hidden_size * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMemcpy for results failed: %s\n", hipGetErrorString(err));
        goto cleanup;
    }
    fprintf(stderr, "DEBUG: Results copied back to host\n");
    
cleanup:
    // Free device memory
    fprintf(stderr, "DEBUG: Freeing device memory\n");
    if (d_x) hipFree(d_x);
    if (d_h0) hipFree(d_h0);
    if (d_h_out) hipFree(d_h_out);
    if (d_a) hipFree(d_a);
    if (d_b) hipFree(d_b);
    min_gru_free_device(&d_linear_z, &d_linear_h);
    fprintf(stderr, "DEBUG: Device memory freed\n");
} 