# Security

`qgemv-roofline` is a set of CUDA microbenchmarks and the scripts that summarise
their output. It runs on hardware you choose and talks to nothing on the network:

- `make` compiles `src/bench.cu` with your local `nvcc`, and `make bench` (and the
  other targets) run that binary on your own GPU and write JSONL into `results/`.
- `scripts/` reads those result files with `json.loads` and writes tables and
  figures. `scripts/energy_probe.py` reads the NVML energy counter and launches the
  benchmark binary through `wsl.exe`.
- The notebooks in `notebooks/` are meant for a hosted GPU runtime (Colab or
  Kaggle). They write the kernels to `/tmp/qgemv`, compile and run them, and, for the
  profiling and energy cells, install `nsight-compute` with `apt-get` and
  `nvidia-ml-py` with `pip` inside that runtime.

There is no server, no account, no credential handling and no data sent anywhere.
The benchmark binary needs no elevated privileges. Treat a notebook from a fork the
way you would a shell script: read it before you run it.

## Reporting a vulnerability

Please report security problems privately, through GitHub's
[private vulnerability reporting](https://github.com/manunicholasjacob/qgemv-roofline/security/advisories/new)
for this repository, rather than in a public issue. Include the commit or release
tag, the command or notebook cell, and what an attacker controls.

In scope: the code in this repository (`src/`, `python/`, `scripts/`, `tests/`,
the notebooks and the generators that write them, the `Makefile` and the CI
workflow). Out of scope: bugs in the CUDA toolkit, the NVIDIA driver, NVML, Triton,
PyTorch, Colab or Kaggle themselves; report those to their maintainers.

Wrong numbers are not a security problem, but they matter here. A result that
disagrees with the ones in `results/` belongs in a public issue with the
**GPU results** template.

This is a single-maintainer project, so expect an acknowledgement within days rather
than hours. Fixes go into `main` and the next tagged release.

## Supported versions

Only the latest tagged release and `main` receive fixes.
