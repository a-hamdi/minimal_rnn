#ifndef PARALLEL_SCAN_H
#define PARALLEL_SCAN_H

#include "rnn_utils.h"

// CPU implementation of the parallel scan algorithm
// a[t] * h[t-1] + b[t] = h[t]
void parallel_scan(int seq_length, int batch_size, int hidden_size,
                  const float* a, const float* b, const float* h0,
                  float* h_out);

// Sequential scan implementation (simpler but not parallel)
void sequential_scan(int seq_length, int batch_size, int hidden_size,
                    const float* a, const float* b, const float* h0,
                    float* h_out);

// Helper structure for parallel scan
typedef struct {
    float* a;  // Coefficient
    float* b;  // Value
} ScanOp;

// Function to combine two scan operations
// op_out = op2 ○ op1 (composition)
// Where (a, b) ○ (c, d) = (a*c, a*d + b)
GPU_CALLABLE void compose_scan_ops(const ScanOp* op1, const ScanOp* op2, ScanOp* op_out, 
                                   int size);

// Apply scan operation to hidden state
// h_out = op.a * h_in + op.b
GPU_CALLABLE void apply_scan_op(const ScanOp* op, const float* h_in, float* h_out, 
                                int size);

// HIP implementation of the parallel scan algorithm
#ifdef HIP_ENABLED
void parallel_scan_hip(int seq_length, int batch_size, int hidden_size,
                       const float* a, const float* b, const float* h0,
                       float* h_out);
#endif

#endif // PARALLEL_SCAN_H 