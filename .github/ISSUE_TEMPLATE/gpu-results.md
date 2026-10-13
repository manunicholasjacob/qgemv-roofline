---
name: GPU results
about: Share the notebook's results file from a GPU that is not in results/ yet
title: "[results] <GPU name, sm_XX>"
labels: results
---

## Where it ran

<!-- Colab, Kaggle, a cloud VM or your own machine. GPU name and compute capability as
nvidia-smi reports them, driver and CUDA versions (the notebook's first cell prints all
three). Was anything else using the GPU? -->

## Results file

<!-- Attach colab_<gpu>_results.json from notebooks/qgemv_portable.ipynb. -->

## Anything that looked wrong?

<!-- A kernel above its own measured ceiling usually means the matrix fit in L2 (see the
Hardware section of the README). A correctness check that failed is worth reporting
even if the timing looks fine. -->
