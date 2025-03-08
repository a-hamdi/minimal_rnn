#include "min_gru.h"
#include "parallel_scan.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA kernel for MinGRU forward pass
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

// CUDA kernel for preparing parallel scan parameters
__global__ void min_gru_extract_scan_params_kernel(const LinearLayer* d_linear_z, 
                                                  const LinearLayer* d_linear_h,
                                                  const float* x, float* a, float* b,
                                                  int seq_length, int hidden_size, int input_size) {
    // Each thread handles one element of the hidden state for one time step
    int t = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (t < seq_length && h < hidden_size) {
        // Get input for current time step
        const float* x_t = x + t * input_size;
        
        // Compute update gate: z_t = sigmoid(Linear_z(x_t))
        float z_t = d_linear_z->bias[h];
        for (int i = 0; i < input_size; i++) {
            z_t += d_linear_z->weights[h * input_size + i] * x_t[i];
        }
        z_t = 1.0f / (1.0f + expf(-z_t)); // sigmoid
        
        // Compute candidate hidden state: h_tilde = Linear_h(x_t)
        float h_tilde = d_linear_h->bias[h];
        for (int i = 0; i < input_size; i++) {
            h_tilde += d_linear_h->weights[h * input_size + i] * x_t[i];
        }
        
        // Compute a_t = 1 - z_t and b_t = z_t * h_tilde
        a[t * hidden_size + h] = 1.0f - z_t;
        b[t * hidden_size + h] = z_t * h_tilde;
    }
}

// Helper for composing scan operations in parallel
__global__ void compose_scan_ops_kernel(const float* a1, const float* b1,
                                       const float* a2, const float* b2,
                                       float* a_out, float* b_out,
                                       int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < hidden_size) {
        // Compute a_out = a2 * a1
        a_out[idx] = a2[idx] * a1[idx];
        
        // Compute b_out = a2 * b1 + b2
        b_out[idx] = a2[idx] * b1[idx] + b2[idx];
    }
}

// Helper for applying scan operation in parallel
__global__ void apply_scan_op_kernel(const float* a, const float* b,
                                    const float* h_in, float* h_out,
                                    int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < hidden_size) {
        // Compute h_out = a * h_in + b
        h_out[idx] = a[idx] * h_in[idx] + b[idx];
    }
}

// CUDA implementation of the parallel scan algorithm
void min_gru_parallel_scan_cuda(int seq_length, int batch_size, int hidden_size,
                               const float* a, const float* b, const float* h0,
                               float* h_out) {
    // Allocate device memory
    float *d_a, *d_b, *d_h0, *d_h_out;
    float *d_temp_a, *d_temp_b;
    
    size_t seq_hidden_size = seq_length * hidden_size * sizeof(float);
    size_t hidden_size_bytes = hidden_size * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, seq_hidden_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, seq_hidden_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_h0, hidden_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_h_out, seq_hidden_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_temp_a, seq_hidden_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_temp_b, seq_hidden_size));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, a, seq_hidden_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, b, seq_hidden_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_h0, h0, hidden_size_bytes, cudaMemcpyHostToDevice));
    
    // For small sequences, use sequential approach on GPU
    if (seq_length <= 8) {
        float *d_h_prev, *d_h_curr;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_h_prev, hidden_size_bytes));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_h_curr, hidden_size_bytes));
        
        // Copy h0 to h_prev
        CHECK_CUDA_ERROR(cudaMemcpy(d_h_prev, d_h0, hidden_size_bytes, cudaMemcpyDeviceToDevice));
        
        // Process each time step
        int threadsPerBlock = 256;
        int blocksPerGrid = (hidden_size + threadsPerBlock - 1) / threadsPerBlock;
        
        for (int t = 0; t < seq_length; t++) {
            // Apply scan operation: h_curr = a_t * h_prev + b_t
            apply_scan_op_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_a + t * hidden_size,
                d_b + t * hidden_size,
                d_h_prev,
                d_h_curr,
                hidden_size
            );
            
            // Copy result to output
            CHECK_CUDA_ERROR(cudaMemcpy(d_h_out + t * hidden_size, d_h_curr, hidden_size_bytes, cudaMemcpyDeviceToDevice));
            
            // Update h_prev for next iteration
            CHECK_CUDA_ERROR(cudaMemcpy(d_h_prev, d_h_curr, hidden_size_bytes, cudaMemcpyDeviceToDevice));
        }
        
        // Free device memory
        CHECK_CUDA_ERROR(cudaFree(d_h_prev));
        CHECK_CUDA_ERROR(cudaFree(d_h_curr));
    } else {
        // Implement parallel scan using a work-efficient tree-based algorithm
        
        // First, compute prefix sums of scan operations
        int threadsPerBlock = 256;
        int blocksPerGrid = (hidden_size + threadsPerBlock - 1) / threadsPerBlock;
        
        // Temporary storage for intermediate results
        float **d_a_temp, **d_b_temp;
        int num_levels = (int)ceil(log2(seq_length));
        
        // Allocate pointers to intermediate results
        d_a_temp = (float**)malloc((num_levels + 1) * sizeof(float*));
        d_b_temp = (float**)malloc((num_levels + 1) * sizeof(float*));
        
        // Level 0 is the input
        d_a_temp[0] = d_a;
        d_b_temp[0] = d_b;
        
        // Allocate memory for intermediate levels
        for (int level = 1; level <= num_levels; level++) {
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a_temp[level], seq_hidden_size));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b_temp[level], seq_hidden_size));
        }
        
        // Up-sweep phase
        for (int level = 0; level < num_levels; level++) {
            int stride = 1 << level;
            int elements = seq_length >> level;
            
            for (int i = 0; i < elements / 2; i++) {
                int left_idx = 2 * i * stride;
                int right_idx = (2 * i + 1) * stride;
                
                if (right_idx < seq_length) {
                    // Compose operations: ops[right_idx] = ops[right_idx] ○ ops[left_idx]
                    compose_scan_ops_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                        d_a_temp[level] + left_idx * hidden_size,
                        d_b_temp[level] + left_idx * hidden_size,
                        d_a_temp[level] + right_idx * hidden_size,
                        d_b_temp[level] + right_idx * hidden_size,
                        d_a_temp[level + 1] + right_idx * hidden_size,
                        d_b_temp[level + 1] + right_idx * hidden_size,
                        hidden_size
                    );
                }
            }
        }
        
        // Allocate device memory for hidden states
        float *d_h_states;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_h_states, (seq_length + 1) * hidden_size_bytes));
        
        // Initialize with h0
        CHECK_CUDA_ERROR(cudaMemcpy(d_h_states, d_h0, hidden_size_bytes, cudaMemcpyDeviceToDevice));
        
        // Compute hidden states using scan operations
        for (int t = 0; t < seq_length; t++) {
            // Apply operation: h_{t+1} = a_t * h_t + b_t
            apply_scan_op_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_a + t * hidden_size,
                d_b + t * hidden_size,
                d_h_states + t * hidden_size,
                d_h_states + (t + 1) * hidden_size,
                hidden_size
            );
            
            // Copy to output
            CHECK_CUDA_ERROR(cudaMemcpy(d_h_out + t * hidden_size, d_h_states + (t + 1) * hidden_size, 
                                        hidden_size_bytes, cudaMemcpyDeviceToDevice));
        }
        
        // Free device memory
        CHECK_CUDA_ERROR(cudaFree(d_h_states));
        
        // Free intermediate memory
        for (int level = 1; level <= num_levels; level++) {
            CHECK_CUDA_ERROR(cudaFree(d_a_temp[level]));
            CHECK_CUDA_ERROR(cudaFree(d_b_temp[level]));
        }
        
        free(d_a_temp);
        free(d_b_temp);
    }
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_h_out, seq_hidden_size, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_h0));
    CHECK_CUDA_ERROR(cudaFree(d_h_out));
    CHECK_CUDA_ERROR(cudaFree(d_temp_a));
    CHECK_CUDA_ERROR(cudaFree(d_temp_b));
}

// Helper function to transfer MinGRU cell to device
void min_gru_to_device(const MinGRUCell* cell, LinearLayer* d_linear_z, LinearLayer* d_linear_h) {
    // Allocate and copy linear_z
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_linear_z->weights, 
                                cell->hidden_size * cell->input_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_linear_z->bias, 
                                cell->hidden_size * sizeof(float)));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_linear_z->weights, cell->linear_z.weights, 
                                cell->hidden_size * cell->input_size * sizeof(float), 
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_linear_z->bias, cell->linear_z.bias, 
                                cell->hidden_size * sizeof(float), 
                                cudaMemcpyHostToDevice));
    
    d_linear_z->input_size = cell->input_size;
    d_linear_z->output_size = cell->hidden_size;
    
    // Allocate and copy linear_h
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_linear_h->weights, 
                                cell->hidden_size * cell->input_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_linear_h->bias, 
                                cell->hidden_size * sizeof(float)));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_linear_h->weights, cell->linear_h.weights, 
                                cell->hidden_size * cell->input_size * sizeof(float), 
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_linear_h->bias, cell->linear_h.bias, 
                                cell->hidden_size * sizeof(float), 
                                cudaMemcpyHostToDevice));
    
    d_linear_h->input_size = cell->input_size;
    d_linear_h->output_size = cell->hidden_size;
}

// Free device memory for MinGRU cell
void min_gru_free_device(LinearLayer* d_linear_z, LinearLayer* d_linear_h) {
    cudaFree(d_linear_z->weights);
    cudaFree(d_linear_z->bias);
    cudaFree(d_linear_h->weights);
    cudaFree(d_linear_h->bias);
}

// Process a full sequence with MinGRU using CUDA
void min_gru_process_sequence_cuda(const MinGRUCell* cell, const float* x, const float* h0,
                                 int seq_length, float* h_out) {
    // Transfer cell to device
    LinearLayer d_linear_z, d_linear_h;
    min_gru_to_device(cell, &d_linear_z, &d_linear_h);
    
    // Allocate device memory for input/output
    float *d_x, *d_h0, *d_h_out;
    float *d_a, *d_b;
    
    size_t input_size_bytes = seq_length * cell->input_size * sizeof(float);
    size_t hidden_size_bytes = cell->hidden_size * sizeof(float);
    size_t seq_hidden_size = seq_length * cell->hidden_size * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_x, input_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_h0, hidden_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_h_out, seq_hidden_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, seq_hidden_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, seq_hidden_size));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, x, input_size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_h0, h0, hidden_size_bytes, cudaMemcpyHostToDevice));
    
    // Extract scan parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (cell->hidden_size + threadsPerBlock - 1) / threadsPerBlock;
    
    dim3 extractBlockDim(threadsPerBlock);
    dim3 extractGridDim(blocksPerGrid, seq_length);
    
    min_gru_extract_scan_params_kernel<<<extractGridDim, extractBlockDim>>>(
        &d_linear_z, &d_linear_h, d_x, d_a, d_b, 
        seq_length, cell->hidden_size, cell->input_size
    );
    
    // Run parallel scan
    min_gru_parallel_scan_cuda(seq_length, 1, cell->hidden_size, d_a, d_b, d_h0, d_h_out);
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_h_out, seq_hidden_size, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_x));
    CHECK_CUDA_ERROR(cudaFree(d_h0));
    CHECK_CUDA_ERROR(cudaFree(d_h_out));
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    
    min_gru_free_device(&d_linear_z, &d_linear_h);
} 