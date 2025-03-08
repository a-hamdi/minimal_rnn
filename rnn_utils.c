#include "rnn_utils.h"
#include <stdlib.h>

// Initialize a linear layer with random weights
void init_linear_layer(LinearLayer* layer, int input_size, int output_size) {
    layer->input_size = input_size;
    layer->output_size = output_size;
    
    // Allocate memory for weights and bias
    layer->weights = (float*)malloc(output_size * input_size * sizeof(float));
    layer->bias = (float*)malloc(output_size * sizeof(float));
    
    // Initialize with small random values
    for (int i = 0; i < output_size * input_size; i++) {
        layer->weights[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }
    
    for (int i = 0; i < output_size; i++) {
        layer->bias[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }
}

// Free memory for linear layer
void free_linear_layer(LinearLayer* layer) {
    free(layer->weights);
    free(layer->bias);
} 