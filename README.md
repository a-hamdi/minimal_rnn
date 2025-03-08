# Minimal RNN Implementation (minGRU and minLSTM)

This repository contains a C and HIP implementation of two minimal recurrent neural network models—minGRU and minLSTM—as described in the paper "Were RNNs All We Needed?". These models are simplified versions of traditional GRUs and LSTMs that remove hidden state dependencies in the computation of their gates, enabling full parallelization during training (via a parallel prefix scan) and a significant reduction in parameter count.

## Key Features

- Pure C implementation of minGRU and minLSTM
- HIP implementation for AMD GPU acceleration
- Support for both sequential mode (for inference) and parallel mode (for training)
- Efficient parallel scan algorithm implementation
- Example code demonstrating usage of both models

## Models

### minGRU (Minimal Gated Recurrent Unit)

The minGRU model simplifies the standard GRU by removing the dependence on the previous hidden state in the gate computations:

- Update Gate: `z_t = σ(Linear_z(x_t))`
- Candidate State: `h_tilde = Linear_h(x_t)`
- Hidden State: `h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h_tilde`

### minLSTM (Minimal Long Short-Term Memory)

The minLSTM model simplifies the standard LSTM in a similar way:

- Forget Gate: `f_t = σ(Linear_f(x_t))`
- Input Gate: `i_t = σ(Linear_i(x_t))`
- Candidate State: `h_tilde = Linear_h(x_t)`
- Gate Normalization: `f'_t = f_t / (f_t + i_t)`, `i'_t = i_t / (f_t + i_t)`
- Hidden State: `h_t = f'_t ⊙ h_{t-1} + i'_t ⊙ h_tilde`

## Parallel Scan Algorithm

Both models can be processed in parallel using a tree-based scan algorithm. The recurrence relation `h_t = a_t ⊙ h_{t-1} + b_t` can be computed efficiently using a parallel scan where:

- For minGRU: `a_t = (1 - z_t)`, `b_t = z_t ⊙ h_tilde`
- For minLSTM: `a_t = f'_t`, `b_t = i'_t ⊙ h_tilde`

## Build Instructions

### Prerequisites

- GCC or compatible C compiler
- ROCm toolkit with HIP (optional, for AMD GPU acceleration)

### Building

To build the CPU-only version:

```bash
make
```

To build with HIP support for AMD GPUs:

```bash
make HIP=1
```

### Cleaning

```bash
make clean
```

## Usage

After building, run the example program:

```bash
./minrnn
```

This will demonstrate both minGRU and minLSTM models running in sequential and parallel modes.

## Example Code

See `main.c` for examples of how to use the minGRU and minLSTM implementations.

## File Structure

- `rnn_utils.h` / `rnn_utils.c`: Common utility functions and data structures
- `parallel_scan.h` / `parallel_scan.c`: Implementation of the parallel scan algorithm
- `min_gru.h` / `min_gru.c`: CPU implementation of minGRU
- `min_lstm.h` / `min_lstm.c`: CPU implementation of minLSTM
- `min_gru_hip.cpp`: HIP implementation of minGRU
- `min_lstm_hip.cpp`: HIP implementation of minLSTM
- `main.c`: Example program demonstrating the usage of the models
- `Makefile`: Build system configuration

## Performance

The parallel scan algorithm enables efficient parallelization of the RNN computation, which is particularly beneficial for long sequences. The HIP implementation further accelerates the computation by leveraging the massive parallelism of AMD GPUs.

## License

This project is provided as open-source software. 