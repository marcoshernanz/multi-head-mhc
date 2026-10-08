# Experiment plan (written before the main results; changes are logged in LOG.md)

## Questions and the comparisons that answer them

All comparisons are **paired**: for a given seed every variant starts from the same main-model weights (the connection parameters are
initialized after them) and sees the same batches in the same order, and all are scored on the same 819k validation tokens
(100 × 8 × 1024 contiguous tokens of FineWeb-Edu validation shard 0). The unit of evidence is the per-seed difference
Δ = loss(variant) − loss(reference); I report its mean, standard error over seeds, t, and how many seeds agree in sign.

| id | question | comparison |
|----|----------|------------|
| H0 | Does mHC beat the plain residual here at all? (sanity / replication) | mHC h1 − residual |
| H1 | Does splitting mHC into channel-group heads (block-diagonal, per-head pre/post/res) help at equal memory and parameters? | A-local h ∈ {2, 4, 8, 16} − mHC h1 |
| H2 | If it helps, is it the structure or the bigger predictor? | A-global h − A-local h; A-static h − mHC-static |
| H3 | Is there an optimum h (MHAR found a U-shape with an optimum at 4–8 heads)? | the h curve of H1 |
| H4 | Block-diagonal heads vs one joint (nh)×(nh) mix (lucidrains `num_fracs`, VWN-style joint index)? | D h4 − A h4 (global and static) |
| H5 | Do heads actually specialize? | probe: head distances of pre/post/res means, token-level std across heads |
| H6 | Is splitting into heads better than spending memory on more copies? | A h4 (n = 4) vs mHC n = 8 |
| H7 | Does it change stability? | LR stress: the best A vs mHC at 2× and 4× the tuned LR |
| H8 | (added after the pilot) Is a local-predictor deficit just slow learning (each head reads 1/h of the stream)? | A-local h with logit gain h − A-local h |

## Decision rules (fixed now)

- Noise floor: the paired seed-to-seed spread measured in the main sweep. An effect "exists" if |mean Δ| > 2 SE and all seeds agree in sign.
- "Viable" for multi-head mHC means: some A variant beats mHC h1 by an effect that exists, at ≤ 10% extra wall time on the same hardware,
  and the effect does not shrink to nothing at the larger scale check.
- A null result is reported as such, with the smallest effect the sweep could have detected (≈ 2 SE).

## Scale and budget

- Main scale: d384, 8 layers, 6 attention heads, SwiGLU 1024, vocab 16,384, sequence 1024, batch 16 × 1024 tokens; 27M parameters.
  Token budget per run fixed after the pilot and the TPU probe (target ≥ 100M tokens, ≈ 4 tokens/param).
- Larger check (only if something is promising): d768, 12 layers on the TPU.
- Seeds: 3 per variant in the main sweep (more if the paired spread is large relative to the effects).
- Compute: TPU v5e-8 (8 single-chip runs in parallel) for the sweep; the 2× T4 GPU quota is shared with other work, so GPU runs
  are kept to pilots and cross-checks.

## Not tested (and why)

- E (mixtures of Sinkhorn heads): no added expressivity (every positive doubly stochastic matrix is Sinkhorn(exp(log M)));
  published in other forms (mHC-lite, BE-HC).
- h = D (per-channel routing): 24·D coefficients per token per sublayer, 6× the stream; impractical without fused kernels.

## Changes after the pilot and the TPU probe (details in LOG.md)

- H8 added (local-predictor gain); answered by exp05 (the gain hurts), so dropped from the main sweep.
- Seeds: 5 per variant in the main sweep instead of 3, because exp05 showed a paired seed spread of 0.028 for local h8
  (15x that of residual vs mHC). Seed-major order: whatever the time limit cuts is the tail of the last seed.
- Main sweep on the TPU in the "streams" layout (same math, checked in float64); the GPU runs used the token layout.

## Changes after exp04 and exp06 (details in LOG.md)

- exp04 left one open question for H2: global heads add predictor parameters (h4 +6.5%, h16 +33%), so they were compared with
  mHC h1 widened to the same parameter count (exp06). Result: equal at h4, the heads far worse at h16.
- The d768 scale check needed that control too, so it compares h1, global h4 and a matched-width h1 (exp07).
- H7: 4x LR broke every variant (exp06), so exp07 tests 2x.

## Changes after exp07, exp08 and exp10 (details in LOG.md)

- exp07's d768 gap came at an untuned LR, and exp10 showed the d384 gain is LR tolerance (zero at 0.5x LR, large at 2x, the
  MLP control breaks like h1 at 2x). So exp09 (d768) sweeps the LR downward first: 7.5e-4 with two seeds, then 1.06e-3 and 2.1e-3;
  3e-3 (2x) is dropped. Decision for the scale condition: it counts as a structural gain at d768 only if global h4 beats its
  control with each variant at its own best LR of the sweep (both seeds agreeing; a second session replicates at those LRs).
- H7 (stability) now has its control, and a second one queued (local h4 at 2x: head structure vs predictor size).
- "h = D per-channel routing: impractical" above holds for token-dependent coefficients. A **static** per-channel read/write
  (SiHC, arXiv 2609.33895, found 09-29) costs only C·n parameters per sublayer; not tested here (static h4/h8 gave nothing).

## Changes after the move to Google Cloud (2026-10-03, written before any exp11-exp14 result)

- exp11 (v6e vs v5e): pass criteria in its jobs.py. If it fails, v6e batches are still valid within themselves (each has its own
  baselines), but no v6e number is put in the same table as a v5e number.
- exp12 (d384 LR grid, sqrt(2) steps 1.5e-3 .. 6e-3, 3 seeds per point, v6e). Read-outs:
  (a) each variant at its own best LR (lowest mean over seeds 0-2) against mHC h1 at its own best, paired by seed;
  (b) global h4 at its best against the MLP control at its best: if this is below zero with an effect that exists, the d384 verdict
      ("heads = their parameters") is wrong and gets corrected;
  (c) the width of each variant's usable range: the LRs within 0.02 of its own best;
  (d) H7b, the mechanism: mHC h1 with every connection parameter's LR x0.25. At 6e-3: within 0.05 of global h4 (or better) means the
      heads' tolerance can be had by slowing the connection, without heads; 0.15 or more worse than global h4 means the split itself
      matters; in between is reported as partial. Global h4 with its connection at x4 at 6e-3 is the reverse check (breaks like mHC
      h1 = consistent with "slower connection dynamics").
- exp13 (d384, 4x the tokens): the question is whether global h4 − control or local h4 − mHC h1 changes with training length; same
  rule (effect exists if |mean| > 2 SE and all 3 seeds agree).
  2026-10-03 23:50 UTC, before any exp13 run: moved to v5e (v6e capacity is the bottleneck; LOG). Read-out unchanged, within
  the batch (it has its own mHC h1, control and residual).
- exp14 (which blocks of the split matter): read-outs in its jobs.py.
- d768 follow-up (after exp09): each variant's best LR from exp09 + exp07 (seeds every LR has), then a replication on v6e with 5 seeds
  at each variant's best LR (mHC h1, global h4, the control, local h4), all in one batch. The scale condition counts as met only if
  global h4 at its best beats the control at its best there with an effect that exists, or local h4 at its best beats mHC h1 at its
  best (equal parameters). A larger model (d1024) only if one of the two holds.
- 2026-10-03 20:50 UTC, before any d768 result of exp09 was in: the d768 follow-up became **exp15**, the same LR grid as exp09
  (7.5e-4, 1.06e-3, 1.5e-3, 2.1e-3; mHC h1, global h4, the control, local h4) on v6e with 3 seeds, then seeds 3-4 at each
  variant's best LR (lowest mean over seeds 0-2). Reason: 4-chip v5e VMs had no spot capacity, so exp09 (2.6 h per d768 run on v5e)
  ends around 03:00 UTC, while a d768 run takes ~35 min on v6e. The scale condition above is read on exp15's 5 seeds at the best
  LRs; exp09 (v5e, 2 seeds, exact pairing with exp07) is the second chip for the same curve. The v6e queue order is now exp11, exp15,
  exp12, exp14, exp13 (exp13, 24k-step runs, last).
- 2026-10-03 21:51 UTC, after exactly one d768 result at a new LR (exp09's `d768-mhc-h1-lr7.5e-4-s0`: 3.938, against 4.146 for
  the same seed at 1.5e-3 in exp07) and before any other: **the d768 grid is extended downward** in sqrt(2) steps, to 5.3e-4 for all
  four variants (exp15: seeds 0-2, first in its queue; exp09: seeds 0-1), and on to 3.75e-4 if any variant's mean at 5.3e-4 beats
  its mean at 7.5e-4 on the same seeds (and so on). Reason: a best LR at the edge of the grid is not a best LR, and the rule "each
  variant at its own best LR" needs every variant's optimum bracketed. The extension is the same for every variant, so it cannot
  favour one; the decision rules above are unchanged.
- 2026-10-04 01:00 UTC, exp15's three lowest LRs complete on seeds 0-2: best LRs by the rule = mHC h1 7.5e-4, global h4 1.06e-3
  (by 0.0007 over 7.5e-4), control 7.5e-4, local h4 7.5e-4; 3.75e-4 not needed (5.3e-4 is worse than 7.5e-4 for every variant).
  Seeds 3-4 at those LRs queued ahead of the rest of 1.5e-3 / 2.1e-3; the scale condition is read on them as written. Secondary,
  outside the rule: global h4 at 7.5e-4 seeds 3-4, for a paired 5-seed comparison of all four variants at one LR.
  `lr_sweep.py`'s best LR now follows the rule literally (mean over --seeds, among the LRs that have them all); before, a half-run
  LR shrank the comparison to the seeds every LR had (seed 0 only for mHC h1, which picked 1.06e-3).
- 2026-10-04 02:10 UTC, **exp15's scale condition read as written: not met**, so no d1024. On 5 seeds at each variant's best LR:
  global h4 (1.06e-3) − control (7.5e-4) = +0.0055 ± 0.0072 two-sample (1/5 better); local h4 − mHC h1 (both 7.5e-4) = +0.0130 ±
  0.0098 (2/5). The rest of exp15 (1.5e-3 / 2.1e-3 seed 2) finishes for the LR curves; then the v6e pool moves to exp12 and exp14.
- 2026-10-04 02:15 UTC, before any exp14 result (prompted by uHC, `notes/literature/recheck_2026-10-04.md`): if exp14's `pre`
  (per-head read only) or `prepost` variant beats the MLP line at its own parameter count at 3e-3 with an effect that exists
  (excess below zero by more than 2 SE of its Δ to mHC h1, all 3 seeds below the line), that variant is run at d768 on v6e at
  7.5e-4, seeds 0-4, against exp15's mHC h1 and control at 7.5e-4 (paired), with the same rule. Otherwise no follow-up: d1024 stays
  off (exp15), and the rest of the budget is not spent for its own sake.
- 2026-10-04 04:25 UTC, **exp12 read as written** (v6e, d384, seeds 0-2): (a) best LR 2.1e-3 for mHC h1, global h4 and the
  control, 3e-3 for local h4 (by 0.003 over 2.1e-3); (b) global h4 − control, both at 2.1e-3, = +0.018 ± 0.004 (0/3), an effect
  that exists but in the direction that does not correct the d384 verdict: at their best LR the heads are worth less than their
  parameters on v6e; (c) no variant with heads has a wider usable range than mHC h1 (control and residual are the widest, 1.5e-3 to
  3e-3); (d) H7b, as written: mHC h1 conn x0.25 − global h4 at 6e-3 = +0.095 ± 0.176 (1/3), "partial", but with that SE it decides
  nothing. Its reverse check contradicts "slower connection dynamics": global h4 conn x4 at 6e-3 is 0.20 ± 0.05 better than global
  h4 (3/3).
- 2026-10-04 04:25 UTC, before any of its runs, **exp16 (exploratory: the hypothesis comes from exp12's data)**: mHC h1 with the
  connection LR x4 at 6e-3 and 2.1e-3, global h4 with it at 2.1e-3, seeds 0-2, v6e, exp12's settings
  (`experiments/exp16_conn_lr/jobs.py`). Read-outs (rule: effect exists if |mean| > 2 SE and all 3 seeds agree):
  (1) at 6e-3, mHC h1 conn x4 − mHC h1, paired. Below −0.10 with an effect that exists = the high-LR tolerance is available
      without heads (a connection-optimizer setting), and the heads' one remaining property is not specific to heads. Within ±0.05,
      or no effect = it is not; then report mHC h1 conn x4 − global h4 conn x4 as how much the split adds at equal settings.
  (2) at 2.1e-3, conn x4 − conn x1 for mHC h1 and for global h4, paired: does the faster connection cost or help at the optimum?
      Any effect that exists is reported as a property of mHC's settings, not of heads.
  (3) at 2.1e-3, global h4 conn x4 − mHC h1 conn x4, paired: the heads at equal connection settings.
  Budget: 9 runs of ~25 min on v6e-1, ~$5. No further follow-up from it, whatever it shows, beyond reporting.
- 2026-10-04 05:20 UTC, after two of exp14's three seeds at 3e-3 (and none of its third): **exp17**, exp14's four part variants at
  2.1e-3 (d384's best LR on v6e, exp12), seeds 0-2, run unconditionally (`experiments/exp17_parts_best_lr/jobs.py`). Reason: exp14
  was fixed at 3e-3 before exp12 showed that 3e-3 is 0.02-0.035 past every variant's best on v6e, where part of a head gain is LR
  tolerance (global h4: −0.015 at 3e-3, −0.004 at 2.1e-3). Read-out: the 02:15 line test at 2.1e-3 (MLP line from exp12's mHC h1 and
  control at 2.1e-3). The 02:15 rule itself is read on 3e-3 as written and decides exp18 (the d768 follow-up, prepared in
  `experiments/exp18_d768_parts/jobs.py` with no variants until the rule is read). If a variant meets the rule at 3e-3 but not at
  2.1e-3, both are reported, and the d768 runs still go ahead as pre-registered: at 7.5e-4 they test it at d768's best LR.
- 2026-10-04 05:50 UTC, **the 02:15 rule read as written** on exp14's 3e-3 runs (seeds 0-2): `pre` excess below the MLP line
  −0.0186 against 2 SE of its Δ 0.0221 (3/3 seeds below the line): not met. `pre,post` (read and write per head, one shared mix)
  −0.0155 against 0.0089 (3/3): met. **exp18 runs pre,post at d768**, v6e, 7.5e-4, seeds 0-4, against exp15's mHC h1 and control
  at 7.5e-4, paired, with the same line test. (`post` alone, outside the rule's scope, would also pass the test: −0.0200 against
  0.0104. It is reported, not followed up.) exp17 (the same variants at 2.1e-3) is the check on whether the 3e-3 gains are LR
  tolerance.
- 2026-10-04 08:41 UTC, **after exp13 (the last batch): no further experiments.** exp13's read-out as written: neither global h4 −
  control nor local h4 − mHC h1 changes with 4x the tokens by an effect that exists (−0.017 ± 0.011 and −0.003 ± 0.005, 2/3 each).
  The one post-hoc pattern (the control's lead over mHC h1 shrinks by 0.018 on every seed, global h4's does not) is not chased:
  it is 1.5 SE for the comparison that matters, a longer run at the fixed LR could not separate structure from LR tolerance, and
  detecting a 0.017 gap against the control's paired seed noise (SD ~0.03) would need about 14 seeds per variant. The scale
  condition for d1024 was not met (exp15), and the scale where other head splits gained (≥ 590M) is out of reach of the $171 left
  (one 600M run at 20 tokens per parameter ≈ 300 v6e chip-hours). Spend: $128.81 of $300; every TPU VM deleted.
- 2026-10-05 11:47 UTC, before any of its runs, **exp19 (stability; `experiments/exp19_stability/jobs.py`)**. Prompted by a proposed
  reframing of the paper: lead with the parameter-matched comparison (no gain) and offer the high-LR tolerance as a stability
  benefit for larger models, citing Wortsman et al. (2023, small-scale proxies) and Qwen's stability results. The tolerance has so
  far been measured only without QK-norm, which current LLMs use and which removes the high-LR instability Wortsman et al. found
  most often, and only up to 6e-3 at d384. Settings as exp12 (v6e, d384, seeds 0-2), plus per-step loss / grad-norm logs and
  diagnostics every 250 steps (largest attention logit and mean attention entropy per layer, sublayer input/output RMS, output z^2).
  Read-outs (an effect exists if |mean| > 2 SE and all 3 seeds agree):
  (S1) Bit-identity: the replays without QK-norm repeat the earlier runs' logged losses exactly. If not, the diagnostics perturb
       the computation; the replays are then read as new runs and this is reported.
  (S2) Mechanism, without QK-norm, descriptive: for each variant at 2.1e-3, 4.2e-3 and 6e-3, the largest attention logit (max over
       layers) and the lowest per-layer attention entropy at steps 500, 1000 and 6000, and z^2. Called "attention-logit growth" if,
       at 6e-3, the variants that fail worst (mHC h1, control) show a largest logit more than twice their 2.1e-3 value at the same
       step and a layer whose mean entropy drops below 0.5 nats.
  (S3) Primary: LR sensitivity as defined by Wortsman et al. (2023), on the 9-point grid 1.5e-3 ... 2.4e-2, per recipe, variant
       and seed: S = mean over the grid of min(final val loss, l0) − l*, with l0 = ln 16384 = 9.704 (a diverged run counts as l0)
       and l* the variant's best mean final loss on the grid. Paired by seed: dS for global h4 − mHC h1, local h4 − mHC h1 and
       global h4 − control. With QK-norm: "the tolerance survives QK-norm" if dS < 0 is an effect for global h4 − mHC h1 or local
       h4 − mHC h1; "QK-norm removes it" if neither is an effect and both are within a third of their size without QK-norm (or of
       opposite sign); anything else is "partial". Without QK-norm, the same dS on the same grid (existing runs up to 6e-3, part C
       above it) is the comparison point.
  (S4) Secondary, with QK-norm: each variant's best LR (lowest mean over seeds 0-2), its usable range (within 0.02 of the best),
       paired differences at each LR above the optimum, and the parameter-matched comparisons at each variant's own best LR:
       local h4 − mHC h1 and global h4 − control (two-sample SE when the LRs differ). These test whether the paper's main
       (null) result holds in a recipe with QK-norm.
  (S5) Spikes, per run from the per-step logs: OLMo 2's spike score (the share of steps whose value lies more than 7 standard
       deviations from the rolling average of the previous 1000 steps; to be checked against the OLMo 2 report before any result
       is read, and logged here if it differs) for training loss and grad norm, and the largest grad norm in the last 80% of
       training (as in exp12). Compared by variant at each LR and recipe, paired by seed.
  (S6) Follow-up rule: only if S3 says "survives", d768 with QK-norm on a sqrt(2) grid 7.5e-4 ... 3e-3 (5 LRs) for mHC h1, global
       h4, local h4 and the control (SwiGLU 2240), seeds 0-2, to see whether the reduction in sensitivity grows with width (Wortsman
       et al.'s scaling argument). Otherwise no d768 runs.
  Budget: 240 d384 runs, ~70 chip-hours of v6e spot, ~$20-45; the controller's budget cap is set to $220 of the $300.
- 2026-10-05 11:51 UTC, exp19 still has no finished run (its VMs are being created), **S5 made precise**. OLMo 2's definition,
  checked in the report (arXiv 2501.00656, §3.2, "Spike score"): the percentage of values in a time series at least seven standard
  deviations from a rolling average of the last 1,000 values, used on training loss and gradient L2 norm. Implemented as written
  (mean and SD of the previous 1,000 steps, so a run's first 1,000 steps are not scored). Two measures from Qwen3.8-Next's stability
  stress test (arXiv 2608.30320, §3.3) are added as descriptive companions, not new read-outs: loss spikes per 10k steps, a step
  whose loss exceeds the median of the 201 steps centred on it by more than 0.1; and the share of steps whose grad norm crosses
  the clipping threshold (1.0 here; the logged grad norm is before clipping, so larger norms do not reach the weights at that size).
- 2026-10-05 11:57 UTC, exp19 still has no finished run, **one descriptive measure added to S4**: the edge of each variant, the
  lowest grid LR whose mean final loss is more than 0.1 above the variant's best, as the location of the cliff (the quantity whose
  scaling Wortsman et al. describe: without QK-norm the LR at which models diverge falls as they grow), next to S3's depth of it.
- 2026-10-05 12:04 UTC, exp19 restarted, no run finished. The first start crashed at step 0 of every run in the new diagnostics
  (on XLA, `amax()` with no dims reduces nothing, so the stacked statistics had mismatched shapes). Fixed with `max()` and float32
  statistics outside autocast; checked on a v6e that runs with and without diagnostics repeat every step's loss and grad norm and
  the final loss exactly (residual 2.1e-3, global h4 with QK-norm 6e-3, mHC 6e-3; 60 steps each). Read-outs unchanged.
- 2026-10-05 16:46 UTC, before any of its runs, **exp20 (d768 with QK-norm; `experiments/exp20_d768_qknorm/jobs.py`), exploratory**.
  Designed after reading most of exp19 (208 of 240 runs in; the QK-norm grid has all its seeds but a few at the top LRs). What has
  been seen: with QK-norm, dS is +0.015 for global h4 − mHC h1 and +0.006 for local h4 − mHC h1, both effects in the direction of
  *more* sensitivity, so S3 will not say "survives" and S6 runs no d768. Seen in the per-step logs, not a pre-registered read-out:
  with QK-norm at d384, mHC h1 and the control cross the clip threshold (1.0) on 0.1-1.3% of the steps in the last 80% of training
  at every LR from 2.1e-3 to 8.5e-3; global h4 on none up to 8.5e-3; local h4 and the residual on almost none. HC-family papers
  (mHC's Fig. 5 included) argue stability from gradient-norm curves, so this is the one stability-like difference that survives
  QK-norm, and it is tested here at d768 rather than reported from the batch in which it was noticed. Exp15's d768 runs (no
  QK-norm, every 25th step logged) show no such difference near their optimum. Settings as exp15 plus --qk-norm 1, per-step logs
  and diagnostics every 250 steps; all five variants (control SwiGLU 2240); seeds 0-2; LRs 5.3e-4, 7.5e-4, 1.06e-3, 1.5e-3, 2.1e-3.
  Read-outs (an effect exists if |mean| > 2 SE and all 3 seeds agree):
  (E1) Gradient norm: per run, the share of steps in the last 80% whose pre-clip grad norm exceeds 1.0, and the largest grad norm
       there. Per LR, paired by seed: global h4 − mHC h1 and local h4 − mHC h1. "Heads smooth the gradient norm at d768" if the
       share's difference is a negative effect for at least one head variant at two or more LRs; "not at d768" if mHC h1's mean
       share is under 0.1% at every LR (nothing to smooth) or no LR shows such an effect; otherwise "partial". OLMo 2's spike
       scores of loss and grad norm alongside, descriptive.
  (E2) The paper's main comparison with QK-norm at d768: each variant's best LR (lowest mean over seeds 0-2 on this grid), then
       local h4 − mHC h1 and global h4 − control at their own best LRs (two-sample SE when the LRs differ), as exp19's S4; and
       mHC h1 − residual. If any variant's best is the grid's lowest LR, 3.75e-4 is added for every variant, seeds 0-2.
  (E3) Descriptive: S over this grid and dS as in S3, beside exp19's dS over d384's lowest five LRs (1.5e-3 ... 6e-3, the same
       4x span). No decision rests on it.
  Budget: 75 runs, ~39 chip-hours of v6e spot, ~$30-40; the controller's budget cap is $250 of the $300.
- 2026-10-05 18:22 UTC, **exp20 stopped with none of its 75 runs finished; it is not run.** In the 1 h 35 min since its start, spot
  v6e capacity let it create one VM (preempted at 17:25 before any run finished) and, at 18:21, two more that were still being
  created; every other attempt failed on quota or capacity. Exp19's VMs over the same hour lived 13-22 minutes before preemption,
  and a 112M run takes about 40 minutes and restarts from step 0 on a new VM, so the batch would spend the remaining budget on
  runs that do not finish. Its controller was stopped and its two VMs deleted (logged in cloud/ledger.jsonl); spend $204.89.
  E1-E3 have no data. What the paper says about 112M rests on exp15 (no QK-norm, every 25th step logged), read with the same late
  clip-crossing measure, descriptively: near the optimum (5.3e-4 ... 1.5e-3) every variant crosses on 0-0.9% of logged late steps
  with no effect between heads and mHC h1; at 2.1e-3, where every variant is far above its best, mHC h1 and the control cross on
  85-88%, global h4 on 38%, local h4 on 28% (paired, 3 of 3 seeds).
- 2026-10-05 19:03 UTC, exp19 at 228 of 240 runs, read-outs unchanged, **the last 12 runs may use one on-demand VM**. They are
  all without QK-norm at 8.5e-3 to 2.4e-2 (seed 2 of most variants, seed 1 of the control at 8.5e-3), and S3's comparison point
  needs them. No spot v6e VM has come up since 18:07. The controller is restarted with `--accel v6e-4 --on-demand-chips 4
  --on-demand-budget 12` (one on-demand v6e-4 at $2.70 per chip-hour, at most $12, besides spot); runs are deterministic on a
  chip type, so the VM type changes the cost, not the result. Total cap still $220.
- 2026-10-05 19:33 UTC, **on-demand v6e is not available to this project**: every on-demand create failed with "User does not have
  permission to submit requests for accelerator type" (no VM was created, nothing spent). The controller is back on spot only,
  with its earlier settings (`--accel v6e-8,v6e-4 --max-chips 32 --budget 220`); the 12 runs wait for spot capacity.
- 2026-10-05 21:05 UTC, **exp19's last 12 runs may use us-east5** (spot v6e at $1.62 per chip-hour, 2.5x the zones used so far).
  No VM has finished a run since 18:06: one came up at 19:58 and was preempted 8 minutes into its runs, and every other create
  failed on capacity. The runs matter for S3: without QK-norm the full-grid S is dominated by whether a run diverges at 8.5e-3,
  and global h4's seed 2 there is still missing. Controller restarted with `--max-price 1.7`; the runs need about 2 chip-hours,
  under $15 with a few preemptions; total cap still $220. Read-outs unchanged.
- 2026-10-05 21:23 UTC, **v6e-1 added as the last fallback size** for exp19's last 12 runs (`--accel v6e-8,v6e-4,v6e-1`):
  every v6e-8 and v6e-4 create failed on capacity in every zone, us-east5 included, from 21:07 to 21:22. Same price per chip;
  read-outs unchanged.
- 2026-10-05 23:31 UTC, **exp19 finalised with 228 of its 240 runs; read-outs.** No spot v6e VM of any size (v6e-8, v6e-4, v6e-1)
  could be created in any zone from 21:07 to 23:30, us-east5 included; one VM came up after 18:07 (19:58, preempted at 20:06).
  Controller stopped, no VM left; spend $205.74. The 12 missing runs are all without QK-norm at 8.5e-3 and above: seed 2 of global
  h4 (8.5e-3, 1.2e-2, 1.7e-2, 2.4e-2), local h4 (8.5e-3, 1.7e-2, 2.4e-2), the control (1.7e-2, 2.4e-2), mHC h1 and the residual
  (2.4e-2), and seed 1 of the control at 8.5e-3. All 36 finished runs without QK-norm at 1.2e-2 and above diverged; at 8.5e-3, 7
  of 12 ended finite. (S1) 42 of 42 replays bit-identical. (S2) Attention-logit growth, by the criterion: at 6e-3 and step 1000
  the largest logit is 2069 for mHC h1 and 2710 for the control (61 and 77 at 2.1e-3), and the lowest layer entropy 0.01 nats.
  (S3) With QK-norm: dS global h4 − mHC h1 +0.0150 ± 0.0019 (0/3, effect), local h4 − mHC h1 +0.0064 ± 0.0026 (0/3, effect),
  global h4 − control +0.0107 ± 0.0068 (0/3). Without QK-norm the full-grid S exists only for complete seeds: global h4 − mHC h1
  −0.013 (s0) and +0.472 (s1); local h4 − mHC h1 −0.067 (s0) and −0.038 (s1); global h4 − control −0.102 (s0). It is dominated by
  whether a run diverges at 8.5e-3 (about 0.47 per divergence). Verdict: **QK-norm removes it**, and the same for every outcome of
  the 12 missing runs (each diverging or ending at 4.5, 5.5 or 7.0: 12288 of 12288 cases; `analysis/stability.py` prints it).
  "Neither is an effect" is read as "dS < 0 is an effect for neither": both with-QK dS are effects in the other direction, which
  the rule's "or of opposite sign" counts as removal. Descriptive, over the five LRs where no run diverges (1.5e-3 ... 6e-3):
  without QK-norm −0.018 ± 0.013 (2/3), −0.028 ± 0.010 (3/3, effect), −0.048 ± 0.012 (3/3, effect); with it +0.004 ± 0.001 (0/3,
  effect), +0.000 ± 0.005 (2/3), +0.007 ± 0.005 (1/3). (S4) With QK-norm, best LRs: mHC h1 and the residual 1.5e-3 (the grid's
  lowest), the control 2.1e-3, global h4 3e-3, local h4 4.2e-3; at each one's best, local h4 − mHC h1 +0.023 ± 0.014 (0/3) and
  global h4 − control +0.023 ± 0.012 (0/3), two-sample; mHC h1 − residual −0.082 ± 0.006 (3/3, effect). Edge 1.7e-2 for all but
  the residual (1.2e-2); without QK-norm 6e-3 for all but the control (4.2e-3). (S5) OLMo 2's spike score of the loss is 0 in
  every run that lasts past step 1000 except mHC h1's seed 1 without QK-norm at 8.5e-3 (0.1% of steps), a run that ends 1.4 above mHC's best; of the grad
  norm at most 0.4% wherever training does not fail. (S6) Not triggered.
