# Experiment E2. `make test` checks the kernels; `make run MACHINE=<name>` writes ../results/e2_<name>.jsonl.
# Two builds: plain -O3 (the plan's setting), and -O3 -ffast-math, which lets the compiler
# vectorize the floating-point sums.
CC ?= cc
CFLAGS = -std=c11 -Wall -Wextra -Werror
MACHINE ?= unnamed
RESULTS = ../results/e2_$(MACHINE).jsonl

all: test_kernels bench_o3 bench_fast

test_kernels: test_kernels.c kernels.c kernels.h
	$(CC) $(CFLAGS) -O1 -o $@ test_kernels.c kernels.c -lm

bench_o3: bench.c kernels.c kernels.h
	$(CC) $(CFLAGS) -O3 -DBUILD_FLAGS='"-O3"' -o $@ bench.c kernels.c -lm

bench_fast: bench.c kernels.c kernels.h
	$(CC) $(CFLAGS) -O3 -ffast-math -DBUILD_FLAGS='"-O3 -ffast-math"' -o $@ bench.c kernels.c -lm

test: test_kernels
	./test_kernels

run: test bench_o3 bench_fast
	./bench_o3 $(MACHINE) > $(RESULTS)
	./bench_fast $(MACHINE) >> $(RESULTS)
	cat $(RESULTS)

clean:
	rm -f test_kernels bench_o3 bench_fast

.PHONY: all test run clean
