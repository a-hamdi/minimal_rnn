#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "min_gru.h"
#include "min_lstm.h"

// Helper function to generate random data
void generate_random_data(float* data, int size, float min_val, float max_val) {
    float range = max_val - min_val;
    for (int i = 0; i < size; i++) {
        data[i] = min_val + ((float)rand() / RAND_MAX) * range;
    }
}

// Helper function to print a vector
void print_vector(const char* name, const float* vec, int size) {
    printf("%s: [", name);
    for (int i = 0; i < size; i++) {
        printf("%.4f", vec[i]);
        if (i < size - 1) printf(", ");
    }
    printf("]\n");
}

// Example for MinGRU
void run_min_gru_example() {
    printf("\n--- MinGRU Example ---\n");
    
    // Parameters
    int input_size = 10;
    int hidden_size = 20;
    int seq_length = 5;
    
    // Initialize the MinGRU cell
    MinGRUCell cell;
    init_min_gru_cell(&cell, input_size, hidden_size);
    
    // Allocate memory for input sequence and hidden states
    float* x = (float*)malloc(seq_length * input_size * sizeof(float));
    float* h0 = (float*)malloc(hidden_size * sizeof(float));
    float* h_out_seq = (float*)malloc(seq_length * hidden_size * sizeof(float));
    float* h_out_par = (float*)malloc(seq_length * hidden_size * sizeof(float));
    
    // Generate random input data
    generate_random_data(x, seq_length * input_size, -1.0f, 1.0f);
    generate_random_data(h0, hidden_size, -1.0f, 1.0f);
    
    // Process the sequence using sequential mode
    printf("Processing with MinGRU (Sequential Mode)...\n");
    min_gru_process_sequence(&cell, x, h0, seq_length, h_out_seq);
    
    // Process the sequence using parallel mode
    printf("Processing with MinGRU (Parallel Mode)...\n");
    float* a = NULL;
    float* b = NULL;
    min_gru_extract_scan_params(&cell, x, seq_length, &a, &b);
    min_gru_parallel_scan(seq_length, 1, hidden_size, a, b, h0, h_out_par);
    
    // Print the first few elements of the final hidden state from both modes
    int print_size = hidden_size < 5 ? hidden_size : 5;
    printf("Final hidden state (first %d elements):\n", print_size);
    print_vector("Sequential", h_out_seq + (seq_length - 1) * hidden_size, print_size);
    print_vector("Parallel", h_out_par + (seq_length - 1) * hidden_size, print_size);
    
    // Check if both modes produce the same result
    float max_diff = 0.0f;
    for (int i = 0; i < seq_length * hidden_size; i++) {
        float diff = fabsf(h_out_seq[i] - h_out_par[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Maximum difference between sequential and parallel modes: %.8f\n", max_diff);
    
    // Free allocated memory
    free(x);
    free(h0);
    free(h_out_seq);
    free(h_out_par);
    free(a);
    free(b);
    free_min_gru_cell(&cell);
}

// Example for MinLSTM
void run_min_lstm_example() {
    printf("\n--- MinLSTM Example ---\n");
    
    // Parameters
    int input_size = 10;
    int hidden_size = 20;
    int seq_length = 5;
    
    // Initialize the MinLSTM cell
    MinLSTMCell cell;
    init_min_lstm_cell(&cell, input_size, hidden_size);
    
    // Allocate memory for input sequence and hidden states
    float* x = (float*)malloc(seq_length * input_size * sizeof(float));
    float* h0 = (float*)malloc(hidden_size * sizeof(float));
    float* h_out_seq = (float*)malloc(seq_length * hidden_size * sizeof(float));
    float* h_out_par = (float*)malloc(seq_length * hidden_size * sizeof(float));
    
    // Generate random input data
    generate_random_data(x, seq_length * input_size, -1.0f, 1.0f);
    generate_random_data(h0, hidden_size, -1.0f, 1.0f);
    
    // Process the sequence using sequential mode
    printf("Processing with MinLSTM (Sequential Mode)...\n");
    min_lstm_process_sequence(&cell, x, h0, seq_length, h_out_seq);
    
    // Process the sequence using parallel mode
    printf("Processing with MinLSTM (Parallel Mode)...\n");
    float* a = NULL;
    float* b = NULL;
    min_lstm_extract_scan_params(&cell, x, seq_length, &a, &b);
    min_lstm_parallel_scan(seq_length, 1, hidden_size, a, b, h0, h_out_par);
    
    // Print the first few elements of the final hidden state from both modes
    int print_size = hidden_size < 5 ? hidden_size : 5;
    printf("Final hidden state (first %d elements):\n", print_size);
    print_vector("Sequential", h_out_seq + (seq_length - 1) * hidden_size, print_size);
    print_vector("Parallel", h_out_par + (seq_length - 1) * hidden_size, print_size);
    
    // Check if both modes produce the same result
    float max_diff = 0.0f;
    for (int i = 0; i < seq_length * hidden_size; i++) {
        float diff = fabsf(h_out_seq[i] - h_out_par[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Maximum difference between sequential and parallel modes: %.8f\n", max_diff);
    
    // Free allocated memory
    free(x);
    free(h0);
    free(h_out_seq);
    free(h_out_par);
    free(a);
    free(b);
    free_min_lstm_cell(&cell);
}

#ifdef HIP_ENABLED
// Example for MinGRU with HIP
void run_min_gru_hip_example() {
    printf("\n--- MinGRU HIP Example (DEBUG MODE) ---\n");
    
    // Parameters - using smaller values for debugging
    int input_size = 10;
    int hidden_size = 20;
    int seq_length = 10; // Smaller sequence for debugging
    
    printf("Parameters: input_size=%d, hidden_size=%d, seq_length=%d\n", 
           input_size, hidden_size, seq_length);
    
    // Initialize the MinGRU cell
    MinGRUCell cell;
    init_min_gru_cell(&cell, input_size, hidden_size);
    printf("MinGRU cell initialized\n");
    
    // Allocate memory for input sequence and hidden states
    float* x = (float*)malloc(seq_length * input_size * sizeof(float));
    float* h0 = (float*)malloc(hidden_size * sizeof(float));
    float* h_out_cpu = (float*)malloc(seq_length * hidden_size * sizeof(float));
    float* h_out_hip = (float*)malloc(seq_length * hidden_size * sizeof(float));
    
    if (!x || !h0 || !h_out_cpu || !h_out_hip) {
        printf("ERROR: Memory allocation failed\n");
        return;
    }
    printf("Memory allocated successfully\n");
    
    // Generate random input data
    generate_random_data(x, seq_length * input_size, -1.0f, 1.0f);
    generate_random_data(h0, hidden_size, -1.0f, 1.0f);
    printf("Random data generated\n");
    
    // Process the sequence using CPU first
    printf("Starting CPU processing...\n");
    clock_t start_cpu = clock();
    min_gru_process_sequence(&cell, x, h0, seq_length, h_out_cpu);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    printf("CPU Processing Time: %.6f seconds\n", cpu_time);
    
    // Process the sequence using HIP
    printf("Starting HIP processing...\n");
    clock_t start_hip = clock();
    min_gru_process_sequence_hip(&cell, x, h0, seq_length, h_out_hip);
    clock_t end_hip = clock();
    double hip_time = ((double)(end_hip - start_hip)) / CLOCKS_PER_SEC;
    printf("HIP Processing Time: %.6f seconds\n", hip_time);
    
    // Check if both modes produce the same result
    float max_diff = 0.0f;
    for (int i = 0; i < seq_length * hidden_size; i++) {
        float diff = fabsf(h_out_cpu[i] - h_out_hip[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Maximum difference between CPU and HIP: %.8f\n", max_diff);
    
    // Free allocated memory
    free(x);
    free(h0);
    free(h_out_cpu);
    free(h_out_hip);
    free_min_gru_cell(&cell);
    printf("Memory freed successfully\n");
}

// Example for MinLSTM with HIP
void run_min_lstm_hip_example() {
    printf("\n--- MinLSTM HIP Example ---\n");
    
    // Parameters
    int input_size = 10;
    int hidden_size = 20;
    int seq_length = 100; // Larger sequence to demonstrate HIP benefit
    
    // Initialize the MinLSTM cell
    MinLSTMCell cell;
    init_min_lstm_cell(&cell, input_size, hidden_size);
    
    // Allocate memory for input sequence and hidden states
    float* x = (float*)malloc(seq_length * input_size * sizeof(float));
    float* h0 = (float*)malloc(hidden_size * sizeof(float));
    float* h_out_cpu = (float*)malloc(seq_length * hidden_size * sizeof(float));
    float* h_out_hip = (float*)malloc(seq_length * hidden_size * sizeof(float));
    
    // Generate random input data
    generate_random_data(x, seq_length * input_size, -1.0f, 1.0f);
    generate_random_data(h0, hidden_size, -1.0f, 1.0f);
    
    // Process the sequence using CPU
    clock_t start_cpu = clock();
    min_lstm_process_sequence(&cell, x, h0, seq_length, h_out_cpu);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    printf("CPU Processing Time: %.6f seconds\n", cpu_time);
    
    // Process the sequence using HIP
    clock_t start_hip = clock();
    min_lstm_process_sequence_hip(&cell, x, h0, seq_length, h_out_hip);
    clock_t end_hip = clock();
    double hip_time = ((double)(end_hip - start_hip)) / CLOCKS_PER_SEC;
    printf("HIP Processing Time: %.6f seconds\n", hip_time);
    
    // Check if both modes produce the same result
    float max_diff = 0.0f;
    for (int i = 0; i < seq_length * hidden_size; i++) {
        float diff = fabsf(h_out_cpu[i] - h_out_hip[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Maximum difference between CPU and HIP: %.8f\n", max_diff);
    
    // Free allocated memory
    free(x);
    free(h0);
    free(h_out_cpu);
    free(h_out_hip);
    free_min_lstm_cell(&cell);
}
#endif

int main() {
    // Seed random number generator
    srand((unsigned int)time(NULL));
    
    // Run CPU examples
    run_min_gru_example();
    run_min_lstm_example();
    
#ifdef HIP_ENABLED
    // Run HIP examples if enabled
    run_min_gru_hip_example();
    run_min_lstm_hip_example();
#endif
    
    return 0;
} 