## Akhilesh Tandur (akhilesh_tandur/)

| Item | Details |
|---|---|
| Tool(s) | Claude (Anthropic), claude.ai |
| What the AI produced | Training and evaluation code for Tasks 1–3 (gpt_char.py, sentiment.py, cyclegan.py, evaluate_local.py), the notebooks, and first drafts of results.md and failure_analysis.md. It also helped with setup and debugging on the lab PC (CUDA build for the RTX 5090, Windows encoding issues). |
| What I did myself | Ran all training and evaluation on the SJSU lab GPU; chose my own hyperparameters (e.g. identity weight 0.3, resize-convolution upsampling and seed 2026 for the CycleGAN); checked the training logs, loss curves and generated samples; reviewed the 20 Task 2 errors and the Task 1 failure cases; submitted to Kaggle; merged my work into the team repo. |
| How I verified it | Causal-mask test in the Task 1 notebook (earlier predictions unchanged when future tokens change); cycle-consistency check in Task 3; McNemar tests and bootstrap confidence intervals in Task 2; cross-checked the reported metrics against the raw logs in reproducibility/. |
| Prompts / conversations | Claude conversation used throughout the lab for code, debugging and write-up drafts. |