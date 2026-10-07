# AI-use disclosure · DATA 266 Lab 1 · Team 40

Our course allows AI tools when their use is disclosed. This file describes how we used them.

## How we used AI

Both of us used **Claude (Anthropic)** as a **coding and learning assistant**:

* **Writing code:** Claude helped us write and structure the code for the three tasks, including first drafts of notebook code,
  model code, the shared evaluation scripts and the configuration files. We read, ran and tested this code, adapted it to our
  setup, and fixed it where it did not work on the lab machines.
* **Learning and understanding:** we asked it to explain concepts (causal masking, cycle-consistency, FID/KID, McNemar's test,
  Cohen's κ, calibration) and why a design choice makes sense, so that we can explain every part of our work ourselves.
* **Debugging and set-up:** we used it to interpret error messages and to work through the GPU-lab set-up (Docker/Jupyter,
  packages, background runs, backups) and the Git workflow.
* **Experiment planning:** for Task 3 it suggested follow-up experiments based on our measured results. We decided which ones to
  run, ran them, and chose the final model with a rule fixed before comparing the runs.
* **Writing:** it helped draft our results, failure analyses and the team report from our own metrics, samples and figures, in
  clear and grammatically correct language. We checked every number, example and figure description against our outputs and
  edited the text.

## What we did ourselves

* Ran **all** training and evaluation on the SJSU GPU lab (RTX 5090) and checked the logs and outputs.
* Made the design and selection decisions, submitted our own Kaggle entries, and did the human audit: both of us rated all
  60 items independently.
* Reviewed the report together and verified that every reported number matches our results files and raw logs.

## What we did not use AI for

* **No numbers or ratings were produced by AI.** All metrics come from our own runs (see `reproducibility/` and
  `task*/akhilesh_tandur/logs/`), and the audit ratings are our own.
* No AI tool generated, edited or selected the images submitted to Kaggle. They are the direct output of our own trained
  CycleGANs; Inception-v3 and LPIPS were used only to *measure* images.
* No pretrained embeddings or language models in Task 2, and no prebuilt Transformer or attention modules in Task 1.

## Akhilesh Tandur (akhilesh_tandur/)

| Item | Details |
|---|---|
| Tool(s) | Claude (Anthropic), claude.ai |
| What the AI produced | Training and evaluation code for Tasks 1–3 (gpt_char.py, sentiment.py, cyclegan.py, evaluate_local.py), the notebooks, and first drafts of results.md and failure_analysis.md. It also helped with setup and debugging on the lab PC (CUDA build for the RTX 5090, Windows encoding issues). |
| What I did myself | Ran all training and evaluation on the SJSU lab GPU; chose my own hyperparameters (e.g. identity weight 0.3, resize-convolution upsampling and seed 2026 for the CycleGAN); checked the training logs, loss curves and generated samples; reviewed the 20 Task 2 errors and the Task 1 failure cases; submitted to Kaggle; merged my work into the team repo. |
| How I verified it | Causal-mask test in the Task 1 notebook (earlier predictions unchanged when future tokens change); cycle-consistency check in Task 3; McNemar tests and bootstrap confidence intervals in Task 2; cross-checked the reported metrics against the raw logs in reproducibility/. |
| Prompts / conversations | Claude conversation used throughout the lab for code, debugging and write-up drafts. |