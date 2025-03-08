#include "parallel_scan.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Combine two scan operations
// op_out = op2 ○ op1 (composition)
// Where (a, b) ○ (c, d) = (a*c, a*d + b)
GPU_CALLABLE void compose_scan_ops(const ScanOp* op1, const ScanOp* op2, ScanOp* op_out, int size) {
    // op_out.a = op2.a * op1.a
    vec_mul(op_out->a, op2->a, op1->a, size);
    
    // temp = op2.a * op1.b
    float* temp = (float*)malloc(size * sizeof(float));
    vec_mul(temp, op2->a, op1->b, size);
    
    // op_out.b = temp + op2.b
    vec_add(op_out->b, temp, op2->b, size);
    
    free(temp);
}

// Apply scan operation to hidden state
// h_out = op.a * h_in + op.b
GPU_CALLABLE void apply_scan_op(const ScanOp* op, const float* h_in, float* h_out, int size) {
    // temp = op.a * h_in
    float* temp = (float*)malloc(size * sizeof(float));
    vec_mul(temp, op->a, h_in, size);
    
    // h_out = temp + op.b
    vec_add(h_out, temp, op->b, size);
    
    free(temp);
}

// CPU implementation of the parallel scan algorithm
// a[t] * h[t-1] + b[t] = h[t]
void parallel_scan(int seq_length, int batch_size, int hidden_size,
                  const float* a, const float* b, const float* h0,
                  float* h_out) {
    // Overall memory requirement is O(log(seq_length) * seq_length * hidden_size)
    
    // Base case: for short sequences, use sequential scan
    if (seq_length <= 4) {
        sequential_scan(seq_length, batch_size, hidden_size, a, b, h0, h_out);
        return;
    }
    
    // Allocate memory for intermediate operations
    ScanOp* ops = (ScanOp*)malloc(seq_length * sizeof(ScanOp));
    for (int i = 0; i < seq_length; i++) {
        ops[i].a = (float*)malloc(hidden_size * sizeof(float));
        ops[i].b = (float*)malloc(hidden_size * sizeof(float));
    }
    
    // Initialize operations with input a and b values
    for (int t = 0; t < seq_length; t++) {
        // Assuming we have a batch of size 1 for simplicity
        const float* a_t = a + t * hidden_size;
        const float* b_t = b + t * hidden_size;
        
        memcpy(ops[t].a, a_t, hidden_size * sizeof(float));
        memcpy(ops[t].b, b_t, hidden_size * sizeof(float));
    }
    
    // Perform parallel scan using tree reduction (up-sweep phase)
    int offset = 1;
    // Building the tree bottom-up
    for (int d = seq_length >> 1; d > 0; d >>= 1) {
        // In each level, we combine pairs of operations at distance 'offset'
        for (int t = 0; t < seq_length; t += 2 * offset) {
            if (t + offset < seq_length) {
                compose_scan_ops(&ops[t], &ops[t + offset], &ops[t + offset], hidden_size);
            }
        }
        offset *= 2;
    }
    
    // Apply operations and compute hidden states (down-sweep phase)
    // Initialize hidden states array
    float** h_states = (float**)malloc((seq_length + 1) * sizeof(float*));
    for (int i = 0; i <= seq_length; i++) {
        h_states[i] = (float*)malloc(hidden_size * sizeof(float));
    }
    
    // h_states[0] = h0
    memcpy(h_states[0], h0, hidden_size * sizeof(float));
    
    // Compute all hidden states using the scan operations
    for (int t = 0; t < seq_length; t++) {
        apply_scan_op(&ops[t], h_states[t], h_states[t + 1], hidden_size);
        
        // Copy to output
        float* h_out_t = h_out + t * hidden_size;
        memcpy(h_out_t, h_states[t + 1], hidden_size * sizeof(float));
    }
    
    // Free allocated memory
    for (int i = 0; i <= seq_length; i++) {
        free(h_states[i]);
    }
    free(h_states);
    
    for (int i = 0; i < seq_length; i++) {
        free(ops[i].a);
        free(ops[i].b);
    }
    free(ops);
}

// Sequential scan for comparison and for small sequence lengths
void sequential_scan(int seq_length, int batch_size, int hidden_size,
                     const float* a, const float* b, const float* h0,
                     float* h_out) {
    // Allocate memory for previous hidden state
    float* h_prev = (float*)malloc(hidden_size * sizeof(float));
    float* h_curr = (float*)malloc(hidden_size * sizeof(float));
    float* temp = (float*)malloc(hidden_size * sizeof(float));
    
    // Initialize h_prev with h0
    memcpy(h_prev, h0, hidden_size * sizeof(float));
    
    // Process each time step sequentially
    for (int t = 0; t < seq_length; t++) {
        // Get a and b for current time step
        const float* a_t = a + t * hidden_size;
        const float* b_t = b + t * hidden_size;
        
        // Compute h_t = a_t * h_prev + b_t
        vec_mul(temp, a_t, h_prev, hidden_size);
        vec_add(h_curr, temp, b_t, hidden_size);
        
        // Store hidden state in output
        float* h_out_t = h_out + t * hidden_size;
        memcpy(h_out_t, h_curr, hidden_size * sizeof(float));
        
        // Update h_prev for next time step
        memcpy(h_prev, h_curr, hidden_size * sizeof(float));
    }
    
    // Free allocated memory
    free(h_prev);
    free(h_curr);
    free(temp);
} 