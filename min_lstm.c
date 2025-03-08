#include "min_lstm.h"
#include "parallel_scan.h"
#include <string.h>

// Initialize a MinLSTM cell
void init_min_lstm_cell(MinLSTMCell* cell, int input_size, int hidden_size) {
    cell->input_size = input_size;
    cell->hidden_size = hidden_size;
    
    // Initialize the linear layers
    init_linear_layer(&cell->linear_f, input_size, hidden_size);
    init_linear_layer(&cell->linear_i, input_size, hidden_size);
    init_linear_layer(&cell->linear_h, input_size, hidden_size);
}

// Free memory for MinLSTM cell
void free_min_lstm_cell(MinLSTMCell* cell) {
    free_linear_layer(&cell->linear_f);
    free_linear_layer(&cell->linear_i);
    free_linear_layer(&cell->linear_h);
}

// MinLSTM forward pass (sequential mode)
// Input: x_t (input vector), h_prev (previous hidden state)
// Output: h_t (new hidden state)
GPU_CALLABLE void min_lstm_forward(const MinLSTMCell* cell, const float* x_t, const float* h_prev, float* h_t) {
    float f_t[cell->hidden_size];        // Forget gate
    float i_t[cell->hidden_size];        // Input gate
    float h_tilde[cell->hidden_size];    // Candidate hidden state
    float gate_sum[cell->hidden_size];   // f_t + i_t
    float f_prime[cell->hidden_size];    // Normalized forget gate
    float i_prime[cell->hidden_size];    // Normalized input gate
    float f_h_prev[cell->hidden_size];   // f_prime * h_prev
    float i_h_tilde[cell->hidden_size];  // i_prime * h_tilde
    
    // Compute forget gate: f_t = sigmoid(Linear_f(x_t))
    linear_forward(&cell->linear_f, x_t, f_t);
    for (int i = 0; i < cell->hidden_size; i++) {
        f_t[i] = sigmoid(f_t[i]);
    }
    
    // Compute input gate: i_t = sigmoid(Linear_i(x_t))
    linear_forward(&cell->linear_i, x_t, i_t);
    for (int i = 0; i < cell->hidden_size; i++) {
        i_t[i] = sigmoid(i_t[i]);
    }
    
    // Compute candidate hidden state: h_tilde = Linear_h(x_t)
    linear_forward(&cell->linear_h, x_t, h_tilde);
    
    // Compute gate sum: gate_sum = f_t + i_t
    vec_add(gate_sum, f_t, i_t, cell->hidden_size);
    
    // Normalize gates: f_prime = f_t / gate_sum, i_prime = i_t / gate_sum
    vec_div_elementwise(f_prime, f_t, gate_sum, cell->hidden_size);
    vec_div_elementwise(i_prime, i_t, gate_sum, cell->hidden_size);
    
    // Compute f_prime * h_prev
    vec_mul(f_h_prev, f_prime, h_prev, cell->hidden_size);
    
    // Compute i_prime * h_tilde
    vec_mul(i_h_tilde, i_prime, h_tilde, cell->hidden_size);
    
    // Compute h_t = f_prime * h_prev + i_prime * h_tilde
    vec_add(h_t, f_h_prev, i_h_tilde, cell->hidden_size);
}

// Process a full sequence with MinLSTM (sequential mode)
// Inputs:
//   - cell: MinLSTM cell
//   - x: input sequence [seq_length x input_size]
//   - h0: initial hidden state [hidden_size]
//   - seq_length: number of time steps
// Output:
//   - h_out: output hidden states [seq_length x hidden_size]
void min_lstm_process_sequence(const MinLSTMCell* cell, const float* x, const float* h0,
                              int seq_length, float* h_out) {
    float h_prev[cell->hidden_size];
    float h_curr[cell->hidden_size];
    
    // Initialize h_prev with h0
    memcpy(h_prev, h0, cell->hidden_size * sizeof(float));
    
    // Process each time step
    for (int t = 0; t < seq_length; t++) {
        // Get input for current time step
        const float* x_t = x + t * cell->input_size;
        
        // Compute hidden state for current time step
        min_lstm_forward(cell, x_t, h_prev, h_curr);
        
        // Store hidden state in output
        float* h_out_t = h_out + t * cell->hidden_size;
        memcpy(h_out_t, h_curr, cell->hidden_size * sizeof(float));
        
        // Update h_prev for next time step
        memcpy(h_prev, h_curr, cell->hidden_size * sizeof(float));
    }
}

// Prepare data for parallel scan by extracting a and b coefficients
// Note: This allocates memory for a and b, caller is responsible for freeing
void min_lstm_extract_scan_params(const MinLSTMCell* cell, const float* x, int seq_length,
                                float** a_out, float** b_out) {
    int hidden_size = cell->hidden_size;
    
    // Allocate memory for a and b arrays
    float* a = (float*)malloc(seq_length * hidden_size * sizeof(float));
    float* b = (float*)malloc(seq_length * hidden_size * sizeof(float));
    
    // Temporary arrays for computation
    float f_t[hidden_size];
    float i_t[hidden_size];
    float h_tilde[hidden_size];
    float gate_sum[hidden_size];
    float f_prime[hidden_size];
    float i_prime[hidden_size];
    float i_h_tilde[hidden_size];
    
    // Compute a and b for each time step
    for (int t = 0; t < seq_length; t++) {
        // Get input for current time step
        const float* x_t = x + t * cell->input_size;
        
        // Compute forget gate: f_t = sigmoid(Linear_f(x_t))
        linear_forward(&cell->linear_f, x_t, f_t);
        for (int i = 0; i < hidden_size; i++) {
            f_t[i] = sigmoid(f_t[i]);
        }
        
        // Compute input gate: i_t = sigmoid(Linear_i(x_t))
        linear_forward(&cell->linear_i, x_t, i_t);
        for (int i = 0; i < hidden_size; i++) {
            i_t[i] = sigmoid(i_t[i]);
        }
        
        // Compute candidate hidden state: h_tilde = Linear_h(x_t)
        linear_forward(&cell->linear_h, x_t, h_tilde);
        
        // Compute gate sum: gate_sum = f_t + i_t
        vec_add(gate_sum, f_t, i_t, hidden_size);
        
        // Normalize gates: f_prime = f_t / gate_sum, i_prime = i_t / gate_sum
        vec_div_elementwise(f_prime, f_t, gate_sum, hidden_size);
        vec_div_elementwise(i_prime, i_t, gate_sum, hidden_size);
        
        // Compute a_t = f_prime
        float* a_t = a + t * hidden_size;
        memcpy(a_t, f_prime, hidden_size * sizeof(float));
        
        // Compute b_t = i_prime * h_tilde
        float* b_t = b + t * hidden_size;
        vec_mul(b_t, i_prime, h_tilde, hidden_size);
    }
    
    *a_out = a;
    *b_out = b;
}

// MinLSTM parallel scan wrapper
void min_lstm_parallel_scan(int seq_length, int batch_size, int hidden_size,
                           const float* a, const float* b, const float* h0,
                           float* h_out) {
    // Call the generic parallel scan implementation
    parallel_scan(seq_length, batch_size, hidden_size, a, b, h0, h_out);
}

// Helper for running MinLSTM in parallel mode on a single sequence
void min_lstm_process_sequence_parallel(const MinLSTMCell* cell, const float* x, const float* h0,
                                       int seq_length, float* h_out) {
    float* a = NULL;
    float* b = NULL;
    
    // Extract a and b coefficients for parallel scan
    min_lstm_extract_scan_params(cell, x, seq_length, &a, &b);
    
    // Run parallel scan
    min_lstm_parallel_scan(seq_length, 1, cell->hidden_size, a, b, h0, h_out);
    
    // Free allocated memory
    free(a);
    free(b);
} 