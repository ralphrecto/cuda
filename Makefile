# Default target name, can be overridden from the command line
# Example: make TARGET=my_program
TARGET ?= gol

# Automatically determine the source file from the target name
SOURCE = $(TARGET).cu

# The CUDA compiler
NVCC = nvcc

# Compiler flags (optional, but good practice)
NVCCFLAGS = -O2

# --- Rules ---

# The default goal is 'all'
.PHONY: all
all: $(TARGET)

# Rule to build the executable
# This links the object file into the final executable
$(TARGET): cuda/$(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o bin/$(TARGET) cuda/$(SOURCE)

# Rule to clean up generated files
.PHONY: clean
clean:
	rm -f $(TARGET)
