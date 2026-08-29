# Build outputs go to $(OUT), which defaults outside the repository, because
# this tree lives in a synced folder and object files do not belong in it.

CUDA    ?= /usr/local/cuda-13.2
NVCC    ?= $(CUDA)/bin/nvcc
ARCH    ?= sm_86
OUT     ?= $(HOME)/kgbuild
PYTHON  ?= python3
REPS    ?= 200

NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Isrc

.PHONY: all bench ceiling triton summary sweep energy notebook test clean

all: $(OUT)/bench

$(OUT)/bench: src/bench.cu src/qgemv_kernels.cuh
	@mkdir -p $(OUT)
	$(NVCC) $(NVCCFLAGS) src/bench.cu -o $@

ceiling: $(OUT)/bench
	@mkdir -p results
	$(OUT)/bench --mode ceiling --reps 50 | tee results/ceiling_rtx3050.jsonl

bench: $(OUT)/bench
	@mkdir -p results
	$(OUT)/bench --mode ceiling --reps 50 > results/ceiling_rtx3050.jsonl
	$(OUT)/bench --M 4096  --K 4096 --reps $(REPS) > results/ladder_4096x4096_rtx3050.jsonl
	$(OUT)/bench --M 11008 --K 4096 --reps $(REPS) > results/ladder_11008x4096_rtx3050.jsonl
	@echo "wrote results/"

sweep: $(OUT)/bench
	@mkdir -p results
	@: > results/sweep_rtx3050.jsonl
	@for K in 1024 2048 4096 8192; do \
	  $(OUT)/bench --M 4096 --K $$K --reps 150 >> results/sweep_rtx3050.jsonl; \
	done
	@echo "wrote results/sweep_rtx3050.jsonl"

triton:
	@mkdir -p results
	$(PYTHON) python/triton_qgemv.py --M 11008 --K 4096 --reps $(REPS) \
	  | tee results/triton_rtx3050.jsonl

# Runs on the Windows host, not here: NVML is only reachable from that side on
# the development machine. See scripts/energy_probe.py.
energy:
	$(PYTHON) scripts/energy_probe.py --sustain 15 --idle 15 \
	  --out results/energy_rtx3050.json

summary:
	$(PYTHON) scripts/summarize.py > docs/RESULTS.md
	@echo "wrote docs/RESULTS.md"

notebook:
	$(PYTHON) scripts/make_portable_notebook.py

test: $(OUT)/bench
	$(PYTHON) tests/test_correctness.py

clean:
	rm -rf $(OUT)
