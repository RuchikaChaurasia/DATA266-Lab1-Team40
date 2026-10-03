# Task 3 — blinded human audit (30 fixed samples, 2 raters)

The 30 audit photos are fixed for the whole team (`task3_gan/common/team_eval_split.json → audit_photos`).
They are never used for training, and each member's notebook saves its translations to `task3_gan/<member>/outputs/audit/`.

1. **Build the pack** (after every member's Task 3 notebook has finished):
   `python task3_gan/audit/make_audit_pack.py`
2. **Rate, blinded.** Each rater opens `task3_gan/audit/pack/audit_sheet.html` and fills in *their own*
   CSV (`ratings_rater1.csv` or `ratings_rater2.csv`) with integers 1–5 for **style, content, artifacts**.
   The page shows only item IDs (Q01…), so raters cannot tell which model produced an image.
   Raters must not open `key/` and must not discuss scores.
3. **Score:** `python task3_gan/audit/score_audit.py` → `audit_results.csv` (per-member means) and
   `audit_agreement.csv` (Cohen's κ quadratic-weighted, plus exact / within-1 agreement).
4. Re-run the last cell of each Task 3 notebook so `results.md` picks up the audit table.

Two raters are required. If a teammate cannot rate, ask a classmate or friend, name them in the report,
and note that rater 1 is also a model author (a possible bias; blinding reduces it).

Reading κ: < 0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, > 0.80 almost perfect
(Landis & Koch, 1977).
