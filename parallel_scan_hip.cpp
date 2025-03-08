#include "parallel_scan.h"
#include <hip/hip_runtime.h>

// Error checking macro
#define CHECK_HIP_ERROR(call) { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error in %s at line %d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
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

// This is a generic implementation that will be used by both minGRU and minLSTM
extern "C" void parallel_scan_hip(int seq_length, int batch_size, int hidden_size,
                       const float* a, const float* b, const float* h0,
                       float* h_out) {
    fprintf(stderr, "DEBUG: Starting parallel_scan_hip\n");
    
    // Allocate device memory
    float *d_a, *d_b, *d_h0, *d_h_out;
    float *d_temp_a, *d_temp_b;
    
    size_t seq_hidden_size = seq_length * hidden_size * sizeof(float);
    size_t hidden_size_bytes = hidden_size * sizeof(float);
    
    fprintf(stderr, "DEBUG: Allocating device memory in parallel_scan_hip\n");
    
    // Since a, b are already on device, we just need to alias them
    d_a = (float*)a;
    d_b = (float*)b;
    d_h0 = (float*)h0;
    d_h_out = (float*)h_out;
    
    fprintf(stderr, "DEBUG: Allocating temporary buffers for intermediate results\n");
    hipError_t err = hipMalloc((void**)&d_temp_a, seq_hidden_size);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc for d_temp_a failed: %s\n", hipGetErrorString(err));
        return;
    }
    
    err = hipMalloc((void**)&d_temp_b, seq_hidden_size);
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipMalloc for d_temp_b failed: %s\n", hipGetErrorString(err));
        hipFree(d_temp_a);
        return;
    }
    
    // Setup kernel launch parameters
    fprintf(stderr, "DEBUG: Setting up kernel launch parameters\n");
    int threadsPerBlock = 256;
    int blocksPerGrid = (hidden_size + threadsPerBlock - 1) / threadsPerBlock;
    
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(blocksPerGrid);
    
    // Initialize the first element in the scan
    fprintf(stderr, "DEBUG: Applying scan operation for first time step\n");
    hipLaunchKernelGGL(apply_scan_op_kernel, gridDim, blockDim, 0, 0,
        d_a, d_b, d_h0, &d_h_out[0], hidden_size
    );
    
    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: Kernel launch for apply_scan_op_kernel failed: %s\n", hipGetErrorString(err));
        goto cleanup;
    }
    
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "ERROR: hipDeviceSynchronize failed after apply_scan_op_kernel: %s\n", hipGetErrorString(err));
        goto cleanup;
    }
    
    // Perform the scan operation for the rest of the sequence
    fprintf(stderr, "DEBUG: Starting scan operation for the rest of the sequence\n");
    for (int t = 1; t < seq_length; t++) {
        fprintf(stderr, "DEBUG: Processing time step %d/%d\n", t, seq_length-1);
        
        // Create aliases for the current and previous elements
        float* a_prev = &d_a[(t-1) * hidden_size];
        float* b_prev = &d_b[(t-1) * hidden_size];
        float* a_curr = &d_a[t * hidden_size];
        float* b_curr = &d_b[t * hidden_size];
        float* h_prev = &d_h_out[(t-1) * hidden_size];
        float* h_curr = &d_h_out[t * hidden_size];
        
        // Compute the composition of the current and previous scan operations
        hipLaunchKernelGGL(compose_scan_ops_kernel, gridDim, blockDim, 0, 0,
            a_prev, b_prev, a_curr, b_curr,
            &d_temp_a[t * hidden_size], &d_temp_b[t * hidden_size],
            hidden_size
        );
        
        err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "ERROR: Kernel launch for compose_scan_ops_kernel failed: %s\n", hipGetErrorString(err));
            goto cleanup;
        }
        
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            fprintf(stderr, "ERROR: hipDeviceSynchronize failed after compose_scan_ops_kernel: %s\n", hipGetErrorString(err));
            goto cleanup;
        }
        
        // Apply the composed operation to the previous hidden state
        hipLaunchKernelGGL(apply_scan_op_kernel, gridDim, blockDim, 0, 0,
            &d_temp_a[t * hidden_size], &d_temp_b[t * hidden_size],
            h0, h_curr, hidden_size
        );
        
        err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "ERROR: Kernel launch for apply_scan_op_kernel (2) failed: %s\n", hipGetErrorString(err));
            goto cleanup;
        }
        
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            fprintf(stderr, "ERROR: hipDeviceSynchronize failed after apply_scan_op_kernel (2): %s\n", hipGetErrorString(err));
            goto cleanup;
        }
    }
    
    fprintf(stderr, "DEBUG: Scan operation completed successfully\n");
    
cleanup:
    // Free temporary device memory
    fprintf(stderr, "DEBUG: Freeing temporary device memory\n");
    hipFree(d_temp_a);
    hipFree(d_temp_b);
    fprintf(stderr, "DEBUG: parallel_scan_hip completed\n");
} 