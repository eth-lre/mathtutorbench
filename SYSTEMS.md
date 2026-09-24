# Systems leaderboard

The model leaderboard answers: *how good a tutor is this model?*

This one answers a different question: *how good a tutor is this whole thing?*
A product, an agentic workflow, a prompt chain, a model with a solver bolted on.
Anything that answers behind an OpenAI-compatible endpoint.

The two are **not comparable**, and are kept in separate tables for that reason.
A system may call several models, retry, verify, and hold information a
leaderboard model never sees. Reading a system score against a model score
rewards exactly the wrong thing.

## How to submit

**1. Expose your system** behind `POST /v1/chat/completions` and `GET /v1/models`.
That is the entire contract. Your prompts, your routing, your tools and your
model choices stay yours: the harness only ever sees a system prompt going out
and a turn coming back.

**2. Fill in a manifest.** Copy `systems/manifest.template.yaml` to
`systems/<your-system>.yaml`. Every field is required.

**3. Run it.**

```bash
python submit_system.py \
    --endpoint http://127.0.0.1:8777/v1 \
    --model my-system \
    --manifest systems/my-system.yaml \
    --score
```

Add `--limit 20` for a smoke run first. The script refuses to emit a submission
whose manifest is incomplete, so you will find out before you pay for a full run.

**4. Open a pull request** with your manifest, your `submission-*.json`, and the
generations. Unlike model submissions, which go through an issue, a system
submission is reviewed as a diff: the manifest is the substance of the claim, and
it should be readable next to the number.

## The four ranked tasks

`scaffolding_generation`, `pedagogy_following`, and their `_hard` variants. They
ask for an open-ended teacher turn, scored by
`eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel` against the human teacher's
turn. Win rate is the share of items where the system's turn outscores the
human's.

The other five tasks ask for a direct answer: solve it, say whether the student
is right, locate the mistake. Many tutoring systems are built never to answer
directly, so those tasks measure the harness more than the tutor. Report them if
you like. They are not ranked.

## Four rules

**Never read a gold field.** `reference_solution` and the other ground-truth
fields are how the scoring works. A system that reads one, anywhere, for any
purpose, is not being measured. This is the only rule with no exception.

**Declare prompt substitution.** `scaffolding_generation` and
`pedagogy_following` load the same dataset and differ *only* by the instruction
given to the tutor. So do the two `_hard` tasks. A system that replaces that
instruction with its own prompt receives byte-identical input on both halves of
a pair: it has run one experiment and reported it twice. That is allowed, and
for most products it is unavoidable, but a claim of "consistent across four
tasks" built on it is inflated.

`submit_system.py` checks this without taking your word for it: it compares the
two runs item by item, and a majority of identical turns is reported as
`task_instruction_replaced`. The check is one-sided. Identical output proves
substitution; different output proves nothing, because a system with sampling
enabled varies anyway.

**Report the no-tools ablation.** Re-run with your external tools and any
private solver disabled, so the system sees only the problem and the dialogue,
as a leaderboard model does. The gap between the two numbers is the part of your
margin that is an information advantage rather than a pedagogical one. It is the
most useful single number a systems submission carries, and it is the one nobody
publishes voluntarily. Subsample if the full set is too expensive, and say so:
subset scores tend to run optimistic.

**Wall-clock is reported, never ranked.** `submit_system.py` times every request
and reports median, p90 and max. Nothing else in this protocol stops a system
from running a ten-minute search across five models per turn, posting a superb
win rate, and being unusable with a student who is stuck at half past nine at
night. The number is measured by the harness rather than self-reported, so it
comes off one clock under one protocol; it still depends on the submitter's
hardware and network, which is exactly why it informs a reader instead of
ordering a table.

## What the harness does not know

It sees an endpoint. It cannot tell how many models you called, whether you
sampled ten turns and kept the best, or what your solver had in hand. That is
the point: the manifest carries what the harness cannot observe, and the pull
request is what makes it reviewable.

So a submission is only worth what its manifest is worth. Write the caveats that
weaken your number. They are the ones a reader cannot reconstruct alone, and
they are what makes the rest of the row believable.
