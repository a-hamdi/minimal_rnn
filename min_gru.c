#include "min_gru.h"
#include "parallel_scan.h"
#include <string.h>

// Initialize a MinGRU cell
void init_min_gru_cell(MinGRUCell* cell, int input_size, int hidden_size) {
    cell->input_size = input_size;
    cell->hidden_size = hidden_size;
    
    // Initialize the linear layers
    init_linear_layer(&cell->linear_z, input_size, hidden_size);
    init_linear_layer(&cell->linear_h, input_size, hidden_size);
}

// Free memory for MinGRU cell
void free_min_gru_cell(MinGRUCell* cell) {
    free_linear_layer(&cell->linear_z);
    free_linear_layer(&cell->linear_h);
}

// MinGRU forward pass (sequential mode)
// Input: x_t (input vector), h_prev (previous hidden state)
// Output: h_t (new hidden state)
GPU_CALLABLE void min_gru_forward(const MinGRUCell* cell, const float* x_t, const float* h_prev, float* h_t) {
    float z_t[cell->hidden_size];  // Update gate
    float h_tilde[cell->hidden_size];  // Candidate hidden state
    float one_minus_z[cell->hidden_size];  // 1 - z_t
    float z_h_tilde[cell->hidden_size];  // z_t * h_tilde
    float one_minus_z_h_prev[cell->hidden_size];  // (1 - z_t) * h_prev
    
    // Compute update gate: z_t = sigmoid(Linear_z(x_t))
    linear_forward(&cell->linear_z, x_t, z_t);
    for (int i = 0; i < cell->hidden_size; i++) {
        z_t[i] = sigmoid(z_t[i]);
    }
    
    // Compute candidate hidden state: h_tilde = Linear_h(x_t)
    linear_forward(&cell->linear_h, x_t, h_tilde);
    
    // Compute (1 - z_t)
    vec_sub_from_scalar(one_minus_z, 1.0f, z_t, cell->hidden_size);
    
    // Compute (1 - z_t) * h_prev
    vec_mul(one_minus_z_h_prev, one_minus_z, h_prev, cell->hidden_size);
    
    // Compute z_t * h_tilde
    vec_mul(z_h_tilde, z_t, h_tilde, cell->hidden_size);
    
    // Compute h_t = (1 - z_t) * h_prev + z_t * h_tilde
    vec_add(h_t, one_minus_z_h_prev, z_h_tilde, cell->hidden_size);
}

// Process a full sequence with MinGRU (sequential mode)
// Inputs:
//   - cell: MinGRU cell
//   - x: input sequence [seq_length x input_size]
//   - h0: initial hidden state [hidden_size]
//   - seq_length: number of time steps
// Output:
//   - h_out: output hidden states [seq_length x hidden_size]
void min_gru_process_sequence(const MinGRUCell* cell, const float* x, const float* h0,
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
        min_gru_forward(cell, x_t, h_prev, h_curr);
        
        // Store hidden state in output
        float* h_out_t = h_out + t * cell->hidden_size;
        memcpy(h_out_t, h_curr, cell->hidden_size * sizeof(float));
        
        // Update h_prev for next time step
        memcpy(h_prev, h_curr, cell->hidden_size * sizeof(float));
    }
}

// Prepare data for parallel scan by extracting a and b coefficients
// Note: This allocates memory for a and b, caller is responsible for freeing
void min_gru_extract_scan_params(const MinGRUCell* cell, const float* x, int seq_length,
                               float** a_out, float** b_out) {
    int hidden_size = cell->hidden_size;
    
    // Allocate memory for a and b arrays
    float* a = (float*)malloc(seq_length * hidden_size * sizeof(float));
    float* b = (float*)malloc(seq_length * hidden_size * sizeof(float));
    
    // Temporary arrays for computation
    float z_t[hidden_size];
    float h_tilde[hidden_size];
    
    // Compute a and b for each time step
    for (int t = 0; t < seq_length; t++) {
        // Get input for current time step
        const float* x_t = x + t * cell->input_size;
        
        // Compute update gate: z_t = sigmoid(Linear_z(x_t))
        linear_forward(&cell->linear_z, x_t, z_t);
        for (int i = 0; i < hidden_size; i++) {
            z_t[i] = sigmoid(z_t[i]);
        }
        
        // Compute candidate hidden state: h_tilde = Linear_h(x_t)
        linear_forward(&cell->linear_h, x_t, h_tilde);
        
        // Compute a_t = (1 - z_t)
        float* a_t = a + t * hidden_size;
        vec_sub_from_scalar(a_t, 1.0f, z_t, hidden_size);
        
        // Compute b_t = z_t * h_tilde
        float* b_t = b + t * hidden_size;
        vec_mul(b_t, z_t, h_tilde, hidden_size);
    }
    
    *a_out = a;
    *b_out = b;
}

// MinGRU parallel scan wrapper
void min_gru_parallel_scan(int seq_length, int batch_size, int hidden_size,
                          const float* a, const float* b, const float* h0,
                          float* h_out) {
    // Call the generic parallel scan implementation
    parallel_scan(seq_length, batch_size, hidden_size, a, b, h0, h_out);
}

// Helper for running MinGRU in parallel mode on a single sequence
void min_gru_process_sequence_parallel(const MinGRUCell* cell, const float* x, const float* h0,
                                      int seq_length, float* h_out) {
    float* a = NULL;
    float* b = NULL;
    
    // Extract a and b coefficients for parallel scan
    min_gru_extract_scan_params(cell, x, seq_length, &a, &b);
    
    // Run parallel scan
    min_gru_parallel_scan(seq_length, 1, cell->hidden_size, a, b, h0, h_out);
    
    // Free allocated memory
    free(a);
    free(b);
} 