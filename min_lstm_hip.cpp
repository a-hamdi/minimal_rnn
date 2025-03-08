#include "min_lstm.h"
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

// HIP kernel for MinLSTM forward pass
__global__ void min_lstm_forward_kernel(const LinearLayer* d_linear_f, const LinearLayer* d_linear_i,
                                       const LinearLayer* d_linear_h, const float* x_t,
                                       const float* h_prev, float* h_t, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < hidden_size) {
        // Temporary variables for this thread's computation
        float f_t = 0.0f;
        float i_t = 0.0f;
        float h_tilde = 0.0f;
        
        // Compute forget gate: f_t = sigmoid(Linear_f(x_t))
        f_t = d_linear_f->bias[idx];
        for (int i = 0; i < d_linear_f->input_size; i++) {
            f_t += d_linear_f->weights[idx * d_linear_f->input_size + i] * x_t[i];
        }
        f_t = 1.0f / (1.0f + expf(-f_t)); // sigmoid
        
        // Compute input gate: i_t = sigmoid(Linear_i(x_t))
        i_t = d_linear_i->bias[idx];
        for (int i = 0; i < d_linear_i->input_size; i++) {
            i_t += d_linear_i->weights[idx * d_linear_i->input_size + i] * x_t[i];
        }
        i_t = 1.0f / (1.0f + expf(-i_t)); // sigmoid
        
        // Compute candidate hidden state: h_tilde = Linear_h(x_t)
        h_tilde = d_linear_h->bias[idx];
        for (int i = 0; i < d_linear_h->input_size; i++) {
            h_tilde += d_linear_h->weights[idx * d_linear_h->input_size + i] * x_t[i];
        }
        
        // Normalize gates: f_prime = f_t / (f_t + i_t), i_prime = i_t / (f_t + i_t)
        float gate_sum = f_t + i_t + 1e-8f; // Add epsilon for numerical stability
        float f_prime = f_t / gate_sum;
        float i_prime = i_t / gate_sum;
        
        // Compute h_t = f_prime * h_prev + i_prime * h_tilde
        h_t[idx] = f_prime * h_prev[idx] + i_prime * h_tilde;
    }
}

// HIP kernel for preparing parallel scan parameters
__global__ void min_lstm_extract_scan_params_kernel(LinearLayer d_linear_f,
                                                  LinearLayer d_linear_i,
                                                  LinearLayer d_linear_h,
                                                  const float* x, float* a, float* b,
                                                  int seq_length, int hidden_size, int input_size) {
    // Each thread handles one element of the hidden state for one time step
    int t = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (t < seq_length && h < hidden_size) {
        // Get input for current time step
        const float* x_t = x + t * input_size;
        
        // Compute forget gate: f_t = sigmoid(Linear_f(x_t))
        float f_t = d_linear_f.bias[h];
        for (int i = 0; i < input_size; i++) {
            f_t += d_linear_f.weights[h * input_size + i] * x_t[i];
        }
        f_t = 1.0f / (1.0f + expf(-f_t)); // sigmoid
        
        // Compute input gate: i_t = sigmoid(Linear_i(x_t))
        float i_t = d_linear_i.bias[h];
        for (int i = 0; i < input_size; i++) {
            i_t += d_linear_i.weights[h * input_size + i] * x_t[i];
        }
        i_t = 1.0f / (1.0f + expf(-i_t)); // sigmoid
        
        // Compute candidate hidden state: h_tilde = Linear_h(x_t)
        float h_tilde = d_linear_h.bias[h];
        for (int i = 0; i < input_size; i++) {
            h_tilde += d_linear_h.weights[h * input_size + i] * x_t[i];
        }
        
        // Compute a_t = f_t and b_t = i_t * h_tilde
        a[t * hidden_size + h] = f_t;
        b[t * hidden_size + h] = i_t * h_tilde;
    }
}

// HIP implementation of the parallel scan algorithm for MinLSTM - just a wrapper around the general one
extern "C" void min_lstm_parallel_scan_hip(int seq_length, int batch_size, int hidden_size,
                               const float* a, const float* b, const float* h0,
                               float* h_out) {
    // This is now a wrapper that calls the shared implementation
    parallel_scan_hip(seq_length, batch_size, hidden_size, a, b, h0, h_out);
}

// Helper function to transfer MinLSTM cell to device
void min_lstm_to_device(const MinLSTMCell* cell, LinearLayer* d_linear_f, LinearLayer* d_linear_i, 
                       LinearLayer* d_linear_h) {
    // Allocate and copy linear_f
    CHECK_HIP_ERROR(hipMalloc((void**)&d_linear_f->weights, 
                             cell->hidden_size * cell->input_size * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc((void**)&d_linear_f->bias, 
                             cell->hidden_size * sizeof(float)));
    
    CHECK_HIP_ERROR(hipMemcpy(d_linear_f->weights, cell->linear_f.weights, 
                             cell->hidden_size * cell->input_size * sizeof(float), 
                             hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_linear_f->bias, cell->linear_f.bias, 
                             cell->hidden_size * sizeof(float), 
                             hipMemcpyHostToDevice));
    
    d_linear_f->input_size = cell->input_size;
    d_linear_f->output_size = cell->hidden_size;
    
    // Allocate and copy linear_i
    CHECK_HIP_ERROR(hipMalloc((void**)&d_linear_i->weights, 
                             cell->hidden_size * cell->input_size * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc((void**)&d_linear_i->bias, 
                             cell->hidden_size * sizeof(float)));
    
    CHECK_HIP_ERROR(hipMemcpy(d_linear_i->weights, cell->linear_i.weights, 
                             cell->hidden_size * cell->input_size * sizeof(float), 
                             hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_linear_i->bias, cell->linear_i.bias, 
                             cell->hidden_size * sizeof(float), 
                             hipMemcpyHostToDevice));
    
    d_linear_i->input_size = cell->input_size;
    d_linear_i->output_size = cell->hidden_size;
    
    // Allocate and copy linear_h
    CHECK_HIP_ERROR(hipMalloc((void**)&d_linear_h->weights, 
                             cell->hidden_size * cell->input_size * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc((void**)&d_linear_h->bias, 
                             cell->hidden_size * sizeof(float)));
    
    CHECK_HIP_ERROR(hipMemcpy(d_linear_h->weights, cell->linear_h.weights, 
                             cell->hidden_size * cell->input_size * sizeof(float), 
                             hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_linear_h->bias, cell->linear_h.bias, 
                             cell->hidden_size * sizeof(float), 
                             hipMemcpyHostToDevice));
    
    d_linear_h->input_size = cell->input_size;
    d_linear_h->output_size = cell->hidden_size;
}

// Helper function to free device memory for a MinLSTM cell
void min_lstm_free_device(LinearLayer* d_linear_f, LinearLayer* d_linear_i, LinearLayer* d_linear_h) {
    fprintf(stderr, "DEBUG: Freeing device memory for MinLSTM cell\n");
    
    // Free linear_f device memory with NULL checks
    if (d_linear_f) {
        if (d_linear_f->weights) {
            fprintf(stderr, "DEBUG: Freeing d_linear_f->weights\n");
            hipError_t err = hipFree(d_linear_f->weights);
            if (err != hipSuccess) {
                fprintf(stderr, "WARNING: hipFree for d_linear_f->weights failed: %s\n", hipGetErrorString(err));
            }
            d_linear_f->weights = NULL;
        }
        
        if (d_linear_f->bias) {
            fprintf(stderr, "DEBUG: Freeing d_linear_f->bias\n");
            hipError_t err = hipFree(d_linear_f->bias);
            if (err != hipSuccess) {
                fprintf(stderr, "WARNING: hipFree for d_linear_f->bias failed: %s\n", hipGetErrorString(err));
            }
            d_linear_f->bias = NULL;
        }
    }
    
    // Free linear_i device memory with NULL checks
    if (d_linear_i) {
        if (d_linear_i->weights) {
            fprintf(stderr, "DEBUG: Freeing d_linear_i->weights\n");
            hipError_t err = hipFree(d_linear_i->weights);
            if (err != hipSuccess) {
                fprintf(stderr, "WARNING: hipFree for d_linear_i->weights failed: %s\n", hipGetErrorString(err));
            }
            d_linear_i->weights = NULL;
        }
        
        if (d_linear_i->bias) {
            fprintf(stderr, "DEBUG: Freeing d_linear_i->bias\n");
            hipError_t err = hipFree(d_linear_i->bias);
            if (err != hipSuccess) {
                fprintf(stderr, "WARNING: hipFree for d_linear_i->bias failed: %s\n", hipGetErrorString(err));
            }
            d_linear_i->bias = NULL;
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
    
    fprintf(stderr, "DEBUG: MinLSTM device memory freed\n");
}

// Process a sequence with MinLSTM using HIP
extern "C" void min_lstm_process_sequence_hip(const MinLSTMCell* cell, const float* x, const float* h0,
                                 int seq_length, float* h_out) {
    fprintf(stderr, "DEBUG: Starting min_lstm_process_sequence_hip\n");
    
    // Setup kernel launch parameters 
    int threadsPerBlock = 256;
    int blocksPerGrid = (cell->hidden_size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 extractBlockDim(threadsPerBlock);
    dim3 extractGridDim(blocksPerGrid, seq_length);
    
    // For debugging with smaller grid
    int simpleBlockSize = 32;
    dim3 debugBlockDim(simpleBlockSize);
    dim3 debugGridDim((cell->hidden_size + simpleBlockSize - 1) / simpleBlockSize, 1);
    
    // Transfer cell to device
    fprintf(stderr, "DEBUG: Transferring cell to device\n");
    LinearLayer d_linear_f, d_linear_i, d_linear_h;
    min_lstm_to_device(cell, &d_linear_f, &d_linear_i, &d_linear_h);
    fprintf(stderr, "DEBUG: Cell transferred to device\n");
    
    // Allocate device memory for input/output
    fprintf(stderr, "DEBUG: Allocating device memory\n");
    float *d_x = NULL, *d_h0 = NULL, *d_h_out = NULL;
    float *d_a = NULL, *d_b = NULL;
    hipError_t err;
    
    size_t input_size_bytes = seq_length * cell->input_size * sizeof(float);
    size_t hidden_size_bytes = cell->hidden_size * sizeof(float);
    size_t seq_hidden_size = seq_length * cell->hidden_size * sizeof(float);
    
    // Allocate and copy with error checking
    fprintf(stderr, "DEBUG: Allocating d_x (%zu bytes)\n", input_size_bytes);
    err = hipMalloc((void**)&d_x, input_size_bytes);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc for d_x failed: %s\n", hipGetErrorString(err));
        goto cleanup;
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
    fprintf(stderr, "DEBUG: Grid dimensions: blocks=%d, threads=%d, seq_length=%d\n", 
            blocksPerGrid, threadsPerBlock, seq_length);
    fprintf(stderr, "DEBUG: Hidden size=%d, Input size=%d\n", cell->hidden_size, cell->input_size);
    
    // Try first with a single timestep
    fprintf(stderr, "DEBUG: Launching min_lstm_extract_scan_params_kernel for first timestep only\n");
    hipLaunchKernelGGL(min_lstm_extract_scan_params_kernel, debugGridDim, debugBlockDim, 0, 0,
        d_linear_f, d_linear_i, d_linear_h, d_x, d_a, d_b, 
        1, cell->hidden_size, cell->input_size
    );
    
    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: Kernel launch failed: %s\n", hipGetErrorString(err));
        goto cleanup;
    }
    
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipDeviceSynchronize failed: %s\n", hipGetErrorString(err));
        goto cleanup;
    }
    fprintf(stderr, "DEBUG: First timestep scan parameters extracted successfully\n");
    
    // Now process all timesteps
    if (seq_length > 1) {
        fprintf(stderr, "DEBUG: Processing all %d timesteps\n", seq_length);
        hipLaunchKernelGGL(min_lstm_extract_scan_params_kernel, extractGridDim, extractBlockDim, 0, 0,
            d_linear_f, d_linear_i, d_linear_h, d_x, d_a, d_b, 
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
    fprintf(stderr, "DEBUG: Running min_lstm_parallel_scan_hip\n");
    min_lstm_parallel_scan_hip(seq_length, 1, cell->hidden_size, d_a, d_b, d_h0, d_h_out);
    fprintf(stderr, "DEBUG: Parallel scan completed\n");
    
    // Copy results back to host
    fprintf(stderr, "DEBUG: Copying results back to host\n");
    err = hipMemcpy(h_out, d_h_out, seq_hidden_size, hipMemcpyDeviceToHost);
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
    min_lstm_free_device(&d_linear_f, &d_linear_i, &d_linear_h);
    fprintf(stderr, "DEBUG: Device memory freed\n");
} 