# Task 1 — Failure analysis (ruchika_chaurasia)

My model is a small character-level GPT (4 layers, 256-dim embeddings, 4 heads, 256-character context, 3.25M
parameters). I used the best checkpoint `bceaa3cfc2f2` (epoch 10, validation loss 0.615, 0.887 bits per character).
I generated text for 5 prompts with greedy decoding and with temperatures 0.7 and 1.0. All examples below are copied
unchanged from `outputs/samples.txt`.

---

## Failure 1: Repetition / looping

*Prompt:* `The little dog was sad because` · greedy decoding (temperature 0) · repeated 4-gram rate **0.456**

```text
... He wished he had not been so brave and he could not find his way back home. He wished he had not been
so brave and he was so sad. He wished he had not been so brave and strong. He wished he had not been so brave
and strong. He wished he had not been so brave and strong. He wished he had not been
```

**Type:** Repetition (the model gets stuck in a loop).

**What I observed:** The story starts fine, but after "He wished he had not been so brave" the model keeps
writing the same sentence until it runs out of length. Greedy decoding always picks the most likely next character,
so once a sentence is in the context, writing it again becomes the most likely choice, and nothing random happens to
break the loop. TinyStories also repeats simple phrases a lot, so the model has learned that repeating is normal. This
happened in almost all my greedy samples (for example "She put it in her pocket and started to play with it" five times).
The numbers show it clearly: the repeated 4-gram rate is **0.343** for greedy decoding but only **0.019** at temperature 0.7.

## Failure 2: Broken spelling / made-up words

*Prompt:* `Once upon a time` · temperature 1.0 · non-word rate **0.047** (highest of all my samples)

```text
Once upon a time there was a boy called Janna. ... She walked to the forest and finally looked up at her sitch,
where she could see a big tree. ... At first she could see it cutter through her fingers a
```

**Type:** Broken spelling / invented words (plus a small gender mistake).

**What I observed:** "sitch" is not a real word and "cutter through her fingers" does not make sense. Another
sample at the same temperature wrote "a big, foury bird". Because my model works with single characters, it does not
know a list of real words; it builds each word letter by letter. At temperature 1.0 it sometimes picks an unlikely letter
in the middle of a word, and after that it just continues the wrong word. The same story also calls Janna "a boy" and
then "she", which shows the model remembers grammar better than facts about a character. With greedy decoding or
temperature 0.7 this almost never happened, so the high temperature is the main cause.

## Failure 3: Loss of coherence (characters change)

*Prompt:* `One day, a girl named Mia found a` · temperature 1.0

```text
One day, a girl named Mia found a sad elephant. She did not want to leave the girl alone. Her mom saw Tim just
screaming and smiling. Mia was scared, but she knew she had to find the girl.

Mia hugged Tim and said, "I know, I know you cried. I crossed down the tree, I crossed the Tim. I see that it has my
cage!" Mia smiled and said, "Yes, Lily. It's okay. You are being kind and successful. You are a good boy."
```

**Type:** Loss of coherence / made-up characters.

**What I observed:** Each sentence is grammatical on its own, but the story does not hold together. "Tim" and
"Lily" suddenly appear even though the story is about Mia, Mia calls someone "Lily" and then "a good boy", and
"I crossed the Tim" uses a name like an object. Tim and Lily are some of the most common names in TinyStories, so when
the model is unsure, it falls back on them. Mia is still inside the 256-character context, so the model did not simply
forget her. My small model is good at local patterns (80.4% next-character accuracy) but not good at keeping track of who
is who. I saw the same thing at temperature 0.7 too ("Billy saw Timmy's friend Billy").

---

## Summary

All three failures come from how the model is built and how the text is generated. **Temperature** is a trade-off:
greedy decoding gives clean words but gets stuck repeating, while temperature 1.0 avoids repetition but creates made-up
words and mixed-up characters. Temperature 0.7 worked best for me (repeated 4-gram rate 0.019, distinct-2 of 0.71).
**Character-level tokens** cause the spelling mistakes, because one bad letter breaks a word. **The small model size and
short context** (4 layers, 256 characters ≈ 45–50 words) are why the model handles spelling and grammar well but not the
whole story. To improve this I would try a bigger model with a longer context, top-k or nucleus sampling or a repetition
penalty, and word or subword tokens instead of single characters.
