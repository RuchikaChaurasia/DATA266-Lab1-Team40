# Task 1 data — TinyStories (shared raw dataset)
Downloaded automatically by the notebook from Hugging Face (`roneneldan/TinyStories`, train split)
and cached by `datasets` (default `~/.cache/huggingface`). Nothing needs to be placed here by hand.

Each member samples their own 100,000 / 10,000 train/validation stories with their own seed. The indices are
saved to `task1_llm/<member>/data_processed/split_indices.json`.

Offline fallback: put `TinyStories-train.txt` (stories separated by `<|endoftext|>`) in this folder and set
`data.local_text_file: task1_llm/data/TinyStories-train.txt` in the member's config.
