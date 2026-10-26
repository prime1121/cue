# Evaluate against Smart Turn

This Rust implementation needs a paired benchmark before making any claim that
it improves on Smart Turn. The built-in evaluator summarizes observed outcomes;
it neither supplies a trained model nor establishes statistical superiority.

## Input schema

The evaluator accepts an array of `SessionResult` records. Each record is one
annotated evaluation segment. IDs must be nonempty and unique. All fields are
required; use explicit null for an absent endpoint. Unknown fields are rejected.

```json
[
  {
    "id": "conversation-017-turn-04",
    "reference_end_ms": 2400,
    "baseline_end_ms": 2900,
    "candidate_end_ms": 2700,
    "baseline_inference_ms": [12.1, 11.8],
    "candidate_inference_ms": [3.2, 3.5]
  },
  {
    "id": "conversation-018-internal-pause-02",
    "reference_end_ms": null,
    "baseline_end_ms": 1200,
    "candidate_end_ms": null,
    "baseline_inference_ms": [12.4],
    "candidate_inference_ms": [3.3]
  }
]
```

All timestamps are unsigned integer milliseconds relative to the beginning of
the same segment. The baseline and candidate fields record each system's **first
emitted endpoint**, including an early endpoint that it later retracts. An
endpoint equal to the reference timestamp counts as correct with zero delay.

`reference_end_ms: null` means an annotator has determined that the segment is
**incomplete throughout the observation window**. It does not mean the label is
unknown, annotation is missing, or the recording was accidentally truncated.
Exclude unknown segments. A prediction on a known-incomplete segment is always
an error. Keep the task definition for waiting and backchannels consistent in
both systems' annotations.

A null model endpoint means the system emitted no endpoint during the common
observation window. On a complete segment this is a miss. On an incomplete
segment this is a correct hold. Set and record a finite, identical observation
window before evaluating; do not stop one model's replay sooner. Include enough
post-reference audio/silence to measure meaningful delayed endpoints. The schema
does not store or verify that window, so retain it in the benchmark manifest.

Inference arrays contain every timed model call, in milliseconds, within the
segment. Finite nonnegative values are required. An empty array is allowed when
measurements are unavailable, but prevents the overall comparison if that model
has no samples across the dataset. Inference measurements are pooled per call,
not averaged per segment. If the baseline performs one expensive inference and
the candidate performs many small calls, these percentiles alone do not measure
the total CPU cost: report call counts, CPU seconds per audio minute, peak memory,
and real-time factor separately.

## Metrics and denominators

For each model:

| Field | Numerator / denominator |
| --- | --- |
| `premature_ends` | Early endpoints on complete segments plus any endpoints on known-incomplete segments / all segments |
| `premature_on_complete` | Early endpoints on complete segments / complete segments |
| `ended_incomplete` | Endpoints on known-incomplete segments / known-incomplete segments |
| `missed_complete` | Complete segments with no endpoint / complete segments |

Rates are fractions in `[0, 1]` with explicit numerator and denominator. A zero
denominator produces a null value, not zero. Complete-segment early endpoints and
misses are distinct, mutually exclusive errors.

`endpoint_delay_ms` includes only correct endpoints and measures prediction time
minus reference time. Early predictions do not improve this distribution with
negative delays. Missing predictions do not disappear from the miss rate.
`inference_ms` measures model-call time independently of endpoint delay. Neither
metric includes assistant generation, network, playback, or TTS latency unless
those are explicitly part of the supplied measurement, which should be avoided
for model-call time.

Both distributions report sample count, p50, and p95 with the **nearest-rank**
definition: sort ascending and choose rank `ceil(p * n)`, counting from 1. There
is no interpolation. Empty distributions produce null percentiles. For two
samples `[10, 20]`, p50 is 10 and p95 is 20.

`candidate_not_worse_on_observed_metrics` is:

- `null` when complete or known-incomplete segments are absent, either model has
  no correct endpoints for delay measurement, or either model has no inference
  samples. Reasons are provided in `comparison_unavailable_reasons`.
- `false` when the required evidence exists but any candidate error rate,
  endpoint-delay percentile, or inference-latency percentile is worse.
- `true` when the required evidence exists and all those metrics are no worse.
  Ties pass. This is a descriptive observation on the provided dataset, not a
  statistical significance test or proof of general superiority.

The comparison cannot reward fewer early cuts if the candidate introduces more
misses, slower endpoint delay, or slower inference percentiles. It also checks
error rates on complete and incomplete segments separately, so their mixture
cannot hide a regression in one of those categories. Delay remains conditional
on successful endpoints, which can differ between systems; inspect paired
examples and report confidence intervals and joint trade-off curves separately.

The crate's `ComparisonMode::Segment` runs a context-aware policy and a policy
with empty context on the same audio and Smart Turn scores. Each policy freezes
after its first endpoint until an explicit reset, and a candidate endpoint does
not erase the baseline's audio history. Only a policy requesting a score receives
that score and its inference timing. When both request the same score, the
inference is shared and the recorded cost is the same measured call for both.
This is a policy comparison around the same pretrained model; it is not an
independent reproduction of Pipecat's complete analyzer or separate CPU timing
trials. The ordinary streaming `Diagnostic` mode resets its baseline at candidate
turn boundaries and is unsuitable for independent multi-turn accuracy claims.

## Benchmark protocol

1. Freeze a held-out test split by **conversation and speaker**, keeping all
   segments from the same conversation or speaker in one split. Where possible,
   hold out recording environments too. Include internal pauses, repairs,
   hesitations, backchannels, explicit waiting, noise, accents, languages, and
   genuine turn completions. Have annotators agree on when a user grants the
   assistant the floor, and retain adjudicated examples of disagreement.
2. Record exact baseline and candidate versions, checkpoint hashes, thresholds,
   VAD settings, audio preprocessing, sample rate, replay chunk size, context
   length, and timeout/fallback behavior. Supply identical audio, with the same
   input timing and observation window. Never give a streaming decision future
   audio, a final transcript, or text unavailable at that point in real time.
3. Sweep thresholds and hangover/pause settings on a separate development split.
   Compare premature and miss rates at matched endpoint-delay budgets; also
   report the full error/latency trade-off. Freeze the operating points before
   opening test results. Do not tune thresholds or lexical rules on the test set.
4. Measure both systems on the same target CPU with the same thread budget,
   precision policy, warm-up procedure, and load. Include any ASR dependency's
   latency, memory, and CPU cost in the system comparison. Report model inference
   time separately from audio-to-endpoint delay and full response latency.
5. Replay held-out sessions and record first endpoints and all inference calls.
   Preserve a manifest with observation windows, hardware, software, split IDs,
   settings, and raw predictions. Run the paired evaluator on those records.
6. Bootstrap confidence intervals by conversation or speaker, respecting the
   split's grouping, and compare paired outcomes. Publish sample counts and
   per-language/noise/session-type slices. Audit changes in misses, late ends,
   and truncated words. A small synthetic fixture only validates the evaluator;
   it is not evidence that this implementation beats Smart Turn.

Any deployment decision should specify an acceptable delay/miss budget and a
target error reduction on representative held-out conversations. A single
aggregate accuracy number cannot establish that a voice assistant behaves
better.
