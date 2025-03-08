#ifndef RNN_UTILS_H
#define RNN_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __HIP_DEVICE_COMPILE__
#define GPU_CALLABLE __host__ __device__
#else
#define GPU_CALLABLE
#endif

// Sigmoid activation function
GPU_CALLABLE inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Element-wise vector operations
GPU_CALLABLE inline void vec_mul(float* result, const float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b[i];
    }
}

GPU_CALLABLE inline void vec_add(float* result, const float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

GPU_CALLABLE inline void vec_scale(float* result, const float* a, float scalar, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * scalar;
    }
}

GPU_CALLABLE inline void vec_sub_from_scalar(float* result, float scalar, const float* a, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = scalar - a[i];
    }
}

GPU_CALLABLE inline void vec_div_elementwise(float* result, const float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] / (b[i] + 1e-8f); // Add epsilon for numerical stability
    }
}

// Linear layer implementation
typedef struct {
    float* weights;  // [output_size x input_size]
    float* bias;     // [output_size]
    int input_size;
    int output_size;
} LinearLayer;

// Initialize a linear layer with random weights
void init_linear_layer(LinearLayer* layer, int input_size, int output_size);

// Free memory for linear layer
void free_linear_layer(LinearLayer* layer);

// Forward pass for linear layer: output = weights * input + bias
GPU_CALLABLE inline void linear_forward(const LinearLayer* layer, const float* input, float* output) {
    for (int o = 0; o < layer->output_size; o++) {
        output[o] = layer->bias[o];
        for (int i = 0; i < layer->input_size; i++) {
            output[o] += layer->weights[o * layer->input_size + i] * input[i];
        }
    }
}

#endif // RNN_UTILS_H 