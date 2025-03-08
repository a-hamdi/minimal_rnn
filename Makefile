CC = gcc
HIPCC = hipcc
CFLAGS = -Wall -Wextra -O3 -std=c99 -I.
HIPCCFLAGS = -O3 -I.
LDFLAGS = -lm

# Build with HIP support if HIP=1 is specified
ifdef HIP
	CFLAGS += -DHIP_ENABLED
	HIP_OBJS = parallel_scan_hip.o min_gru_hip.o min_lstm_hip.o
	LINKER = $(HIPCC)
else
	HIP_OBJS =
	LINKER = $(CC)
endif

# Object files
CPU_OBJS = rnn_utils.o parallel_scan.o min_gru.o min_lstm.o main.o
OBJS = $(CPU_OBJS) $(HIP_OBJS)

# Targets
all: minrnn

minrnn: $(OBJS)
	$(LINKER) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(HIPCC) $(HIPCCFLAGS) -c $< -o $@

clean:
	rm -f minrnn $(OBJS)

# Dependencies
rnn_utils.o: rnn_utils.h
parallel_scan.o: parallel_scan.h rnn_utils.h
min_gru.o: min_gru.h rnn_utils.h parallel_scan.h
min_lstm.o: min_lstm.h rnn_utils.h parallel_scan.h
parallel_scan_hip.o: parallel_scan.h rnn_utils.h
min_gru_hip.o: min_gru.h rnn_utils.h parallel_scan.h
min_lstm_hip.o: min_lstm.h rnn_utils.h parallel_scan.h
main.o: min_gru.h min_lstm.h

.PHONY: all clean 