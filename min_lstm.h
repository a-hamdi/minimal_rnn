#ifndef MIN_LSTM_H
#define MIN_LSTM_H

#include "rnn_utils.h"

// MinLSTM cell structure
typedef struct {
    LinearLayer linear_f;  // Forget gate linear layer
    LinearLayer linear_i;  // Input gate linear layer
    LinearLayer linear_h;  // Candidate hidden state linear layer
    int hidden_size;
    int input_size;
} MinLSTMCell;

// Initialize a MinLSTM cell
void init_min_lstm_cell(MinLSTMCell* cell, int input_size, int hidden_size);

// Free memory for MinLSTM cell
void free_min_lstm_cell(MinLSTMCell* cell);

// MinLSTM forward pass (sequential mode)
// Input: x_t (input vector), h_prev (previous hidden state)
// Output: h_t (new hidden state)
GPU_CALLABLE void min_lstm_forward(const MinLSTMCell* cell, const float* x_t, const float* h_prev, float* h_t);

// Extract a and b parameters for parallel scan
// Note: This allocates memory for a and b, caller is responsible for freeing
void min_lstm_extract_scan_params(const MinLSTMCell* cell, const float* x, int seq_length,
                               float** a_out, float** b_out);

// Parallel scan for MinLSTM
// Inputs:
//   - seq_length: number of time steps
//   - batch_size: number of sequences in batch
//   - hidden_size: dimension of hidden states
//   - a: coefficients array [seq_length x batch_size x hidden_size]
//   - b: values array [seq_length x batch_size x hidden_size]
//   - h0: initial hidden state [batch_size x hidden_size]
// Output:
//   - h_out: output hidden states [seq_length x batch_size x hidden_size]
void min_lstm_parallel_scan(int seq_length, int batch_size, int hidden_size,
                           const float* a, const float* b, const float* h0,
                           float* h_out);

// HIP implementation of MinLSTM parallel scan
#ifdef HIP_ENABLED
void min_lstm_parallel_scan_hip(int seq_length, int batch_size, int hidden_size,
                              const float* a, const float* b, const float* h0,
                              float* h_out);

// Process a full sequence with MinLSTM using HIP
void min_lstm_process_sequence_hip(const MinLSTMCell* cell, const float* x, const float* h0,
                                int seq_length, float* h_out);
#endif

// Process a full sequence with MinLSTM (sequential mode)
// Inputs:
//   - cell: MinLSTM cell
//   - x: input sequence [seq_length x input_size]
//   - h0: initial hidden state [hidden_size]
//   - seq_length: number of time steps
// Output:
//   - h_out: output hidden states [seq_length x hidden_size]
void min_lstm_process_sequence(const MinLSTMCell* cell, const float* x, const float* h0,
                             int seq_length, float* h_out);

// Helper for running MinLSTM in parallel mode on a single sequence
void min_lstm_process_sequence_parallel(const MinLSTMCell* cell, const float* x, const float* h0,
                                      int seq_length, float* h_out);

#endif // MIN_LSTM_H 