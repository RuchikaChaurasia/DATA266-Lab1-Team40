# Task 2 — Error review (ruchika_chaurasia)

Reviewed model: **A_exp2_bilstm_attn** (my best model, checkpoint `ff1c3543fe50`), official Yelp Polarity test set
(38,000 reviews), threshold 0.5. Full table with the review texts: `outputs/error_review_20.csv`.
The notebook picked 20 errors automatically: 5 confident false positives, 5 confident false negatives, 5 near-threshold
errors and 5 errors from the slice with the highest error rate ("has contrast word", 6.4% errors). The notebook's
keyword-based error types were only a first guess. I read every review and assigned the final type myself (table below).

Labels: Yelp Polarity marks 1–2 star reviews as negative and 4–5 star reviews as positive, so the "true" label is the
star rating the reviewer gave, not a judgement of the text.

## Confident false positives (true: negative, model: P(pos) ≈ 1.0)

| # | P(pos) | Review (short) | Final error type | Note |
|---|---|---|---|---|
| 1 | 1.000 | "Wow love the place and everything is very clean and new! Great place to come and relax worth a try!" | **Label noise** | The text is completely positive; the low star rating does not match it. The model is actually right about the text. |
| 2 | 1.000 | Foodland: "The bakery here is tops ... Huge brownies ... The best thing I found is the individual pineapple upside-down cakes" | **Mixed sentiment** | Most of the review praises single items, so positive words dominate, but the overall rating was low. |
| 3 | 1.000 | Historic bar: "I love learning about the history ... The food is frozen ... Imitation crab & lots of filling" | **Mixed sentiment** | Positive about the history and the bartender, negative about the food; the model counted the positive part more. |
| 4 | 1.000 | "20 years for me and sad to see them go ... consistently great fresh fish and great service" | **Change over time (past vs now)** | The praise is about the restaurant's past; the negative part is that it closed. The model only sees the praise words. |
| 5 | 1.000 | "About average ... very tasty but a little over cooked ... more for show than content ... I'd recommend Ruth Chris, LG's ..." | **Mixed sentiment** | Words like "tasty" and "fantastic" fool the model, but the verdict ("average", "recommend other places") is negative. |

## Confident false negatives (true: positive, model: P(pos) ≈ 0.0)

| # | P(pos) | Review (short) | Final error type | Note |
|---|---|---|---|---|
| 1 | 0.000 | Aquarium: "Don't get me wrong: This place is TINY! Narrow walk ways ... Even the exhibits are small." | **Mixed sentiment** | The review starts with complaints; the positive overall opinion is weaker in the text than the criticism. |
| 2 | 0.000 | "i wasn't in an objective frame of mind ... this place left a smile on my face ... sawdust? really? WHY?" | **Negation + rhetorical questions** | "wasn't", "really?" and "WHY?" read as negative to the model, but the review is playful and positive. |
| 3 | 0.000 | "The food is crap ... it really is horrible ... probably the worst nachos I've ever had" | **Label noise** | The text is clearly negative but labelled positive. The model's prediction fits the text. |
| 4 | 0.000 | "EDIT: They really did change the service ... Horrible service ... an altercation with a server" | **Change over time (edited review)** | The reviewer edited the text to be negative but the star rating still reflects the old, positive visit. |
| 5 | 0.000 | "There is no denying that the food is great... but I am back to concerned about the service ... wait was closer to 25 mins" | **Mixed sentiment** | Good food but a long list of service problems; the rating stayed positive while most of the text is negative. |

## Near-threshold errors (P(pos) ≈ 0.5)

| # | True | P(pos) | Review (short) | Final error type | Note |
|---|---|---|---|---|---|
| 1 | neg | 0.500 | "The price is a little high for me. They don't give enough bean sprouts ... broth is a little too sweet" | **Weak / hedged sentiment** | Mild complaints softened by "a little"; no strong negative words, so the model is unsure. |
| 2 | neg | 0.500 | Hospital: "The Dr was extremely rude ... she said Google it" | **Domain shift** | This is a hospital review, not a restaurant. Apart from "rude", the words are medical, which the model rarely saw. (The notebook guessed "sarcasm", which is wrong.) |
| 3 | pos | 0.500 | "no more waffle fries, SOOO disappointing ... :( It's not a big enough price difference to make me boycott their kickass French dip" | **Mixed sentiment** | Disappointment and praise in the same review. My preprocessing also removed the emoticon ":(" with the punctuation. |
| 4 | pos | 0.499 | "This Is what every domed ball park should strive to be. Nuff said." | **Implicit sentiment / too short** | Only 13 words and no typical sentiment words; the praise is implied ("should strive to be"), and "Nuff" is slang outside the vocabulary. |
| 5 | neg | 0.501 | Lux coffee (643 words): "coffee service that is definitely above Starbucks, but I will admit struggles in certain areas ..." | **Long review + truncation** | Very long and mixed. With head + tail truncation at 256 tokens, the middle of the review is cut off. |

## Slice-specific errors ("has contrast word", the worst slice)

| # | True | P(pos) | Review (short) | Final error type | Note |
|---|---|---|---|---|---|
| 1 | pos | 0.000 | "UPDATED. My initial very frustrated and dramatic review read as follows: Billing practices are at best negligent ..." | **Change over time (updated review)** | The reviewer quotes their old negative review before saying the problem was fixed; the quoted complaint dominates. |
| 2 | neg | 1.000 | "nearly 5-years ... time for a quick update, and a slight downgrade. - Parking lot ... - Produce quality ..." | **Mixed sentiment** | A long-time customer listing complaints in neutral language; years of loyalty and mild wording read as positive. |
| 3 | pos | 0.000 | "So much better since they changed owners ... old owners, it was terrible ... It was horrible. Now its much better." | **Change over time (past vs now)** | Most of the text describes the bad past; only the end describes the good present. |
| 4 | pos | 0.000 | Tailors: "Up a few steps then your in sort of place ... A constantly busy place, brides in getting alterations done ..." | **Implicit sentiment (descriptive text)** | A neutral description with almost no opinion words; the positive rating is never stated directly. |
| 5 | neg | 1.000 | "NOTE: This was a 4-star review, but the food quality and ESPECIALLY customer service have gone down the tubes ... That sub was so freaking good!!!" | **Change over time (edited review)** | The old enthusiastic text is still in the review; the new negative rating is explained only in a short note. |

## Error-type summary (my final types)

| Error type | Count |
|---|---|
| Mixed sentiment (positive and negative in one review) | 7 |
| Change over time (updated / edited review, past vs now) | 5 |
| Weak, hedged or implicit sentiment (incl. very short reviews) | 3 |
| Label noise (text clearly contradicts the star rating) | 2 |
| Negation + rhetorical questions | 1 |
| Domain shift (non-restaurant review) | 1 |
| Long review + truncation | 1 |

**Main finding:** 12 of the 20 errors (mixed sentiment + change over time) come from reviews where the opinion
**changes inside the review**. The BiLSTM reads the whole review as one sequence and then averages it with attention,
so when both opinions appear, the side with more emotional words wins, even if the writer's final verdict is the other
one. In 2 cases (label noise) the model was arguably right and the label was wrong, so not every "error" is a model mistake.

## Proposed testable fix

**Fix:** make the model look at the review **sentence by sentence**. Split each review into sentences, encode each
sentence with the BiLSTM, and add a second attention layer over the sentences (a hierarchical attention network,
Yang et al., 2016), so the model can learn that sentences such as "Now it's much better", "EDIT: ...", or the last
sentence after "but" carry the final verdict.

**How to test it:** retrain with the same data, seed and hyperparameters and compare with the current BiLSTM on the
same test set. The fix works if (1) macro-F1 on the **"has contrast word"** slice improves from **0.9352** by at least
1 point, and (2) a paired McNemar test on that slice is significant (p < 0.05), while (3) overall accuracy does not drop
below **0.9425**.

**A smaller, cheaper fix** for one error I found: keep emoticons such as ":(" and ":)" and "!" as tokens instead of
deleting them with the punctuation, and check whether the error rate on short reviews (currently 6.04%) goes down.
