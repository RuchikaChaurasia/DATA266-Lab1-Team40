## Joint analysis (write in your own words, using the numbers in the tables above)

### Task 1
TODO — Which lineup reached lower validation loss / bits-per-character, and why (depth vs context length,
tied vs untied head, cosine vs linear schedule)? Compare the generalisation gap and training stability
(spikes, gradient norms). Link to the failure analyses: which failures are shared, and which are specific to one model?

### Task 2
TODO — Did the experimental models beat each baseline, and was the difference statistically significant
(McNemar p, overlapping CIs)? Compare lemmatisation vs stemming and head_tail vs head truncation. Which slices
(negation, contrast, long reviews) are hardest for all models? Is the most accurate model also the best calibrated (ECE/Brier)?
What does the extra accuracy cost in time and memory?

### Task 3
TODO — ResNet/LSGAN vs U-Net/vanilla: which has better FID/KID in each direction, better content preservation
(LPIPS, content cosine, cycle L1), and better human-audit scores? Do the automatic metrics agree with the human
raters? How reliable is the audit (kappa)? What do the loss curves say about stability? Report Kaggle scores/rank.

### Overall lessons
TODO — 3–5 sentences on what the team learned across the three tasks.
