CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CXXFLAGS ?=

GENCODE_FLAGS ?= \
	-gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_86,code=sm_86 \
	-gencode arch=compute_89,code=sm_89 \
	-gencode arch=compute_90,code=sm_90

NVCCFLAGS ?= -O3 -std=c++17 -lineinfo $(GENCODE_FLAGS) $(CXXFLAGS)

SRC_DIR := src
BIN_DIR := bin

.PHONY: all clean vector_add reduction matmul softmax flash_attention run-vector-add run-reduction run-matmul run-softmax run-flash-attention help

all: vector_add reduction matmul softmax flash_attention

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

vector_add: $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/vector_add.cu -o $(BIN_DIR)/vector_add

reduction: $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/reduction.cu -o $(BIN_DIR)/reduction

matmul: $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/matmul.cu -o $(BIN_DIR)/matmul

softmax: $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/softmax.cu -o $(BIN_DIR)/softmax

flash_attention: $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/flash_attention.cu -o $(BIN_DIR)/flash_attention

run-vector-add: vector_add
	./$(BIN_DIR)/vector_add $(ARGS)

run-reduction: reduction
	./$(BIN_DIR)/reduction $(ARGS)

run-matmul: matmul
	./$(BIN_DIR)/matmul $(ARGS)

run-softmax: softmax
	./$(BIN_DIR)/softmax $(ARGS)

run-flash-attention: flash_attention
	./$(BIN_DIR)/flash_attention $(ARGS)

clean:
	rm -rf $(BIN_DIR) build

help:
	@printf "Targets:\n"
	@printf "  make vector_add\n"
	@printf "  make reduction\n"
	@printf "  make matmul\n"
	@printf "  make softmax\n"
	@printf "  make flash_attention\n"
	@printf "  make run-vector-add ARGS=\"--n 67108864 --iterations 100\"\n"
	@printf "  make run-reduction ARGS=\"--n 16777216 --version v4 --iterations 50\"\n"
	@printf "  make run-matmul ARGS=\"--m 1024 --n 1024 --k 1024 --version v2 --iterations 20\"\n"
	@printf "  make run-softmax ARGS=\"--m 4096 --n 4096 --version v2 --iterations 50\"\n"
	@printf "  make run-flash-attention ARGS=\"--n 2048 --version flash --iterations 50\"\n"

