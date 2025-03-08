#ifndef MIN_GRU_H
#define MIN_GRU_H

#include "rnn_utils.h"

// MinGRU cell structure
typedef struct {
    LinearLayer linear_z;  // Update gate linear layer
    LinearLayer linear_h;  // Candidate hidden state linear layer
    int hidden_size;
    int input_size;
} MinGRUCell;

// Initialize a MinGRU cell
void init_min_gru_cell(MinGRUCell* cell, int input_size, int hidden_size);

// Free memory for MinGRU cell
void free_min_gru_cell(MinGRUCell* cell);

// MinGRU forward pass (sequential mode)
// Input: x_t (input vector), h_prev (previous hidden state)
// Output: h_t (new hidden state)
GPU_CALLABLE void min_gru_forward(const MinGRUCell* cell, const float* x_t, const float* h_prev, float* h_t);

// Extract a and b parameters for parallel scan
// Note: This allocates memory for a and b, caller is responsible for freeing
void min_gru_extract_scan_params(const MinGRUCell* cell, const float* x, int seq_length,
                              float** a_out, float** b_out);

// Parallel scan for MinGRU
// Inputs:
//   - seq_length: number of time steps
//   - batch_size: number of sequences in batch
//   - hidden_size: dimension of hidden states
//   - a: coefficients array [seq_length x batch_size x hidden_size]
//   - b: values array [seq_length x batch_size x hidden_size]
//   - h0: initial hidden state [batch_size x hidden_size]
// Output:
//   - h_out: output hidden states [seq_length x batch_size x hidden_size]
void min_gru_parallel_scan(int seq_length, int batch_size, int hidden_size,
                          const float* a, const float* b, const float* h0,
                          float* h_out);

// HIP implementation of MinGRU parallel scan
#ifdef HIP_ENABLED
void min_gru_parallel_scan_hip(int seq_length, int batch_size, int hidden_size,
                              const float* a, const float* b, const float* h0,
                              float* h_out);

// Process a full sequence with MinGRU using HIP
void min_gru_process_sequence_hip(const MinGRUCell* cell, const float* x, const float* h0,
                               int seq_length, float* h_out);
#endif

// Process a full sequence with MinGRU (sequential mode)
// Inputs:
//   - cell: MinGRU cell
//   - x: input sequence [seq_length x input_size]
//   - h0: initial hidden state [hidden_size]
//   - seq_length: number of time steps
// Output:
//   - h_out: output hidden states [seq_length x hidden_size]
void min_gru_process_sequence(const MinGRUCell* cell, const float* x, const float* h0,
                            int seq_length, float* h_out);

// Helper for running MinGRU in parallel mode on a single sequence
void min_gru_process_sequence_parallel(const MinGRUCell* cell, const float* x, const float* h0,
                                     int seq_length, float* h_out);

#endif // MIN_GRU_H 