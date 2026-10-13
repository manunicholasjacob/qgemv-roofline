# Build outputs go to $(OUT), which defaults outside the repository, because
# this tree lives in a synced folder and object files do not belong in it.

CUDA    ?= /usr/local/cuda-13.2
NVCC    ?= $(CUDA)/bin/nvcc
ARCH    ?= sm_86
OUT     ?= $(HOME)/kgbuild
PYTHON  ?= python3
REPS    ?= 200
# Tag for result file names. The shipped results are the RTX 3050 Laptop's; set
# DEVICE (and ARCH) when you run this on another card so they are not overwritten,
# e.g. make bench ARCH=sm_89 DEVICE=rtx4090
DEVICE  ?= rtx3050

NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Isrc

.PHONY: all bench ceiling triton summary sweep energy notebook test clean

all: $(OUT)/bench

$(OUT)/bench: src/bench.cu src/qgemv_kernels.cuh
	@mkdir -p $(OUT)
	$(NVCC) $(NVCCFLAGS) src/bench.cu -o $@

ceiling: $(OUT)/bench
	@mkdir -p results
	$(OUT)/bench --mode ceiling --reps 50 | tee results/ceiling_$(DEVICE).jsonl

bench: $(OUT)/bench
	@mkdir -p results
	$(OUT)/bench --mode ceiling --reps 50 > results/ceiling_$(DEVICE).jsonl
	$(OUT)/bench --M 4096  --K 4096 --reps $(REPS) > results/ladder_4096x4096_$(DEVICE).jsonl
	$(OUT)/bench --M 11008 --K 4096 --reps $(REPS) > results/ladder_11008x4096_$(DEVICE).jsonl
	@echo "wrote results/"

sweep: $(OUT)/bench
	@mkdir -p results
	@: > results/sweep_$(DEVICE).jsonl
	@for K in 1024 2048 4096 8192; do \
	  $(OUT)/bench --M 4096 --K $$K --reps 150 >> results/sweep_$(DEVICE).jsonl; \
	done
	@echo "wrote results/sweep_$(DEVICE).jsonl"

triton:
	@mkdir -p results
	$(PYTHON) python/triton_qgemv.py --M 11008 --K 4096 --reps $(REPS) \
	  | tee results/triton_$(DEVICE).jsonl

# Runs on the Windows host, not here: NVML is only reachable from that side on
# the development machine. See scripts/energy_probe.py.
energy:
	$(PYTHON) scripts/energy_probe.py --sustain 15 --idle 15 \
	  --out results/energy_$(DEVICE).json

summary:
	$(PYTHON) scripts/summarize.py > docs/RESULTS.md
	@echo "wrote docs/RESULTS.md"

notebook:
	$(PYTHON) scripts/make_portable_notebook.py

test: $(OUT)/bench
	$(PYTHON) tests/test_correctness.py

clean:
	rm -rf $(OUT)
