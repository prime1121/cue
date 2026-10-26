//! Paired endpoint evaluation on a shared audio clock and observation window.
//! A null reference means known-incomplete; any endpoint on that row is premature.
//! Delay covers correct endpoints only and must be read alongside miss/error rates.

use std::collections::HashSet;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// One paired result for a complete or known-incomplete segment.
///
/// Timestamps record the first endpoint in segment-relative milliseconds. A null
/// prediction means no endpoint within the window. Include all model calls in
/// the latency vectors, even when no endpoint was emitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionResult {
    pub id: String,
    #[serde(deserialize_with = "required_endpoint")]
    pub reference_end_ms: Option<u64>,
    #[serde(deserialize_with = "required_endpoint")]
    pub baseline_end_ms: Option<u64>,
    #[serde(deserialize_with = "required_endpoint")]
    pub candidate_end_ms: Option<u64>,
    pub baseline_inference_ms: Vec<f64>,
    pub candidate_inference_ms: Vec<f64>,
}

// Missing JSON annotations must not silently become known-incomplete labels.
// Explicit null is allowed; an absent field is rejected by serde.
fn required_endpoint<'de, D>(deserializer: D) -> std::result::Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<u64>::deserialize(deserializer)
}

/// A fraction in [0, 1], with an explicit denominator. No eligible rows -> null.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Rate {
    pub numerator: usize,
    pub denominator: usize,
    pub value: Option<f64>,
}

impl Rate {
    fn new(numerator: usize, denominator: usize) -> Self {
        Self {
            numerator,
            denominator,
            value: (denominator != 0).then(|| numerator as f64 / denominator as f64),
        }
    }
}

/// Nearest-rank quantiles, ceil(p * n), without interpolation; empty means null.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Distribution {
    pub count: usize,
    pub p50: Option<f64>,
    pub p95: Option<f64>,
}

impl Distribution {
    fn from_samples(mut values: Vec<f64>) -> Self {
        values.sort_by(f64::total_cmp);
        let percentile = |percent: usize| {
            if values.is_empty() {
                return None;
            }
            // Compute ceil(n * percent / 100) without overflowing n * percent.
            let rank =
                (values.len() / 100) * percent + ((values.len() % 100) * percent).div_ceil(100);
            Some(values[rank - 1])
        };
        Self {
            count: values.len(),
            p50: percentile(50),
            p95: percentile(95),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelMetrics {
    /// Complete segments with an endpoint at or after the reference timestamp.
    pub correct_endpoints: usize,
    /// All premature endpoints / all segments (complete plus incomplete).
    pub premature_ends: Rate,
    /// Complete segments ended early / all complete segments.
    pub premature_on_complete: Rate,
    /// Known-incomplete segments ended at all / all known-incomplete segments.
    pub ended_incomplete: Rate,
    /// Complete segments with no emitted endpoint / all complete segments.
    pub missed_complete: Rate,
    /// Prediction minus reference, only for correct complete-segment endpoints.
    pub endpoint_delay_ms: Distribution,
    /// All supplied inference calls pooled across segments, independently of delay.
    pub inference_ms: Distribution,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ComparisonReport {
    pub sessions: usize,
    pub complete_sessions: usize,
    pub incomplete_sessions: usize,
    pub baseline: ModelMetrics,
    pub candidate: ModelMetrics,
    /// No worse on all reported rates and p50/p95 latencies, with ties allowed.
    /// Null when either segment class, correct endpoints, or timings are absent.
    pub candidate_not_worse_on_observed_metrics: Option<bool>,
    pub comparison_unavailable_reasons: Vec<String>,
}

/// Evaluate the same annotated segments for both systems.
///
/// Validates IDs and latencies. The caller ensures annotation quality, equal
/// windows, hardware, and complete timing traces; see docs/EVALUATION.md.
pub fn evaluate(sessions: &[SessionResult]) -> Result<ComparisonReport> {
    if sessions.is_empty() {
        bail!("evaluation dataset must contain at least one segment");
    }
    let mut ids = HashSet::with_capacity(sessions.len());
    for session in sessions {
        if session.id.trim().is_empty() {
            bail!("evaluation segment ID must not be empty");
        }
        if !ids.insert(session.id.as_str()) {
            bail!("duplicate evaluation segment ID: {}", session.id);
        }
        for (model, latencies) in [
            ("baseline", &session.baseline_inference_ms),
            ("candidate", &session.candidate_inference_ms),
        ] {
            for (index, latency) in latencies.iter().enumerate() {
                if !latency.is_finite() || *latency < 0.0 {
                    bail!(
                        "segment {}: {} inference latency at index {} must be finite and nonnegative",
                        session.id,
                        model,
                        index
                    );
                }
            }
        }
    }

    let complete = sessions
        .iter()
        .filter(|session| session.reference_end_ms.is_some())
        .count();
    let incomplete = sessions.len() - complete;
    let baseline = model_metrics(sessions, complete, false);
    let candidate = model_metrics(sessions, complete, true);
    let mut unavailable = Vec::new();
    if complete == 0 {
        unavailable.push("no annotated complete segments".to_owned());
    }
    if incomplete == 0 {
        unavailable.push("no annotated known-incomplete segments".to_owned());
    }
    for (name, metrics) in [("baseline", &baseline), ("candidate", &candidate)] {
        if metrics.endpoint_delay_ms.count == 0 {
            unavailable.push(format!(
                "{} has no correct endpoints for delay measurement",
                name
            ));
        }
        if metrics.inference_ms.count == 0 {
            unavailable.push(format!("{} has no inference latency measurements", name));
        }
    }

    let not_worse = if unavailable.is_empty() {
        // Paired evaluation guarantees identical denominators for both models.
        Some(
            candidate.premature_ends.numerator <= baseline.premature_ends.numerator
                && candidate.premature_on_complete.numerator
                    <= baseline.premature_on_complete.numerator
                && candidate.ended_incomplete.numerator <= baseline.ended_incomplete.numerator
                && candidate.missed_complete.numerator <= baseline.missed_complete.numerator
                && distribution_not_worse(
                    &candidate.endpoint_delay_ms,
                    &baseline.endpoint_delay_ms,
                )
                && distribution_not_worse(&candidate.inference_ms, &baseline.inference_ms),
        )
    } else {
        None
    };

    Ok(ComparisonReport {
        sessions: sessions.len(),
        complete_sessions: complete,
        incomplete_sessions: incomplete,
        baseline,
        candidate,
        candidate_not_worse_on_observed_metrics: not_worse,
        comparison_unavailable_reasons: unavailable,
    })
}

fn distribution_not_worse(candidate: &Distribution, baseline: &Distribution) -> bool {
    match (candidate.p50, candidate.p95, baseline.p50, baseline.p95) {
        (Some(c50), Some(c95), Some(b50), Some(b95)) => c50 <= b50 && c95 <= b95,
        _ => false,
    }
}

fn model_metrics(sessions: &[SessionResult], complete: usize, candidate: bool) -> ModelMetrics {
    let mut early = 0;
    let mut false_end = 0;
    let mut missed = 0;
    let mut delays = Vec::new();
    let mut inference = Vec::new();

    for session in sessions {
        let (endpoint, latencies) = if candidate {
            (session.candidate_end_ms, &session.candidate_inference_ms)
        } else {
            (session.baseline_end_ms, &session.baseline_inference_ms)
        };
        inference.extend_from_slice(latencies);
        match (session.reference_end_ms, endpoint) {
            (Some(reference), Some(predicted)) if predicted >= reference => {
                delays.push((predicted - reference) as f64);
            }
            (Some(_), Some(_)) => early += 1,
            (Some(_), None) => missed += 1,
            (None, Some(_)) => false_end += 1,
            (None, None) => {}
        }
    }

    ModelMetrics {
        correct_endpoints: delays.len(),
        premature_ends: Rate::new(early + false_end, sessions.len()),
        premature_on_complete: Rate::new(early, complete),
        ended_incomplete: Rate::new(false_end, sessions.len() - complete),
        missed_complete: Rate::new(missed, complete),
        endpoint_delay_ms: Distribution::from_samples(delays),
        inference_ms: Distribution::from_samples(inference),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(
        id: &str,
        reference: Option<u64>,
        baseline: Option<u64>,
        candidate: Option<u64>,
    ) -> SessionResult {
        SessionResult {
            id: id.to_owned(),
            reference_end_ms: reference,
            baseline_end_ms: baseline,
            candidate_end_ms: candidate,
            baseline_inference_ms: vec![2.0, 4.0],
            candidate_inference_ms: vec![1.0, 3.0],
        }
    }

    #[test]
    fn errors_use_explicit_eligible_denominators() {
        let report = evaluate(&[
            segment("early", Some(100), Some(99), Some(100)),
            segment("missing", Some(100), None, Some(150)),
            segment("incomplete", None, Some(1000), None),
            segment("hold", None, None, None),
        ])
        .unwrap();
        assert_eq!(report.baseline.premature_ends, Rate::new(2, 4));
        assert_eq!(report.baseline.premature_on_complete, Rate::new(1, 2));
        assert_eq!(report.baseline.ended_incomplete, Rate::new(1, 2));
        assert_eq!(report.baseline.missed_complete, Rate::new(1, 2));
        assert_eq!(report.candidate.correct_endpoints, 2);
        assert_eq!(report.candidate.endpoint_delay_ms.p50, Some(0.0));
        assert_eq!(report.candidate.endpoint_delay_ms.p95, Some(50.0));
        assert_eq!(report.candidate_not_worse_on_observed_metrics, None);
    }

    #[test]
    fn zero_denominators_and_empty_distributions_are_null() {
        let report = evaluate(&[segment("hold", None, None, None)]).unwrap();
        assert_eq!(report.baseline.missed_complete.value, None);
        assert_eq!(report.baseline.premature_on_complete.value, None);
        assert_eq!(report.baseline.ended_incomplete.value, Some(0.0));
        assert_eq!(report.baseline.endpoint_delay_ms.count, 0);
        assert_eq!(report.baseline.endpoint_delay_ms.p50, None);
        assert_eq!(report.candidate_not_worse_on_observed_metrics, None);

        let report = evaluate(&[segment("complete", Some(10), Some(10), Some(10))]).unwrap();
        assert_eq!(report.baseline.ended_incomplete.value, None);
        assert_eq!(report.candidate_not_worse_on_observed_metrics, None);
    }

    #[test]
    fn fewer_cuts_with_more_misses_does_not_pass() {
        let report = evaluate(&[
            segment("miss", Some(100), Some(90), None),
            segment("correct", Some(100), Some(100), Some(100)),
            segment("hold", None, None, None),
        ])
        .unwrap();
        assert!(
            report.candidate.premature_ends.numerator < report.baseline.premature_ends.numerator
        );
        assert_eq!(report.candidate.missed_complete.value, Some(0.5));
        assert_eq!(report.candidate_not_worse_on_observed_metrics, Some(false));
    }

    #[test]
    fn abstaining_on_every_complete_segment_is_not_safer() {
        let report = evaluate(&[
            segment("complete", Some(100), Some(90), None),
            segment("hold", None, Some(1), None),
        ])
        .unwrap();
        assert_eq!(report.candidate.missed_complete.value, Some(1.0));
        assert_eq!(report.candidate_not_worse_on_observed_metrics, None);
    }

    #[test]
    fn fewer_cuts_with_slower_endpoints_does_not_pass() {
        let report = evaluate(&[
            segment("correct", Some(100), Some(100), Some(500)),
            segment("hold", None, Some(1), None),
        ])
        .unwrap();
        assert_eq!(report.candidate_not_worse_on_observed_metrics, Some(false));
    }

    #[test]
    fn slower_model_inference_does_not_pass_even_with_faster_endpoints() {
        let mut complete = segment("complete", Some(100), Some(200), Some(100));
        complete.candidate_inference_ms = vec![100.0, 100.0];
        let report = evaluate(&[complete, segment("hold", None, None, None)]).unwrap();
        assert_eq!(report.candidate_not_worse_on_observed_metrics, Some(false));
    }

    #[test]
    fn observed_nonregression_and_ties_pass_when_evidence_exists() {
        let mut complete = segment("complete", Some(100), Some(100), Some(100));
        complete.candidate_inference_ms = complete.baseline_inference_ms.clone();
        let mut hold = segment("hold", None, None, None);
        hold.candidate_inference_ms = hold.baseline_inference_ms.clone();
        let report = evaluate(&[complete, hold]).unwrap();
        assert_eq!(report.candidate_not_worse_on_observed_metrics, Some(true));
        assert!(report.comparison_unavailable_reasons.is_empty());
    }

    #[test]
    fn quantiles_are_deterministic_nearest_rank() {
        let samples: Vec<f64> = (1..=20).rev().map(f64::from).collect();
        let distribution = Distribution::from_samples(samples);
        assert_eq!(distribution.count, 20);
        assert_eq!(distribution.p50, Some(10.0));
        assert_eq!(distribution.p95, Some(19.0));
        let singleton = Distribution::from_samples(vec![7.0]);
        assert_eq!(singleton.p50, Some(7.0));
        assert_eq!(singleton.p95, Some(7.0));
        let pair = Distribution::from_samples(vec![20.0, 10.0]);
        assert_eq!(pair.p50, Some(10.0));
        assert_eq!(pair.p95, Some(20.0));
    }

    #[test]
    fn missing_inference_measurements_prevent_comparison() {
        let mut complete = segment("complete", Some(10), Some(10), Some(10));
        complete.candidate_inference_ms.clear();
        let mut hold = segment("hold", None, None, None);
        hold.candidate_inference_ms.clear();
        let report = evaluate(&[complete, hold]).unwrap();
        assert_eq!(report.candidate.inference_ms.p50, None);
        assert_eq!(report.candidate_not_worse_on_observed_metrics, None);
    }

    #[test]
    fn invalid_datasets_are_rejected() {
        assert!(evaluate(&[]).is_err());
        let duplicate = segment("same", None, None, None);
        assert!(evaluate(&[duplicate.clone(), duplicate]).is_err());
        assert!(evaluate(&[segment(" ", None, None, None)]).is_err());
        for invalid in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut row = segment("bad-baseline", None, None, None);
            row.baseline_inference_ms.push(invalid);
            assert!(evaluate(&[row]).is_err());
            let mut row = segment("bad-candidate", None, None, None);
            row.candidate_inference_ms.push(invalid);
            assert!(evaluate(&[row]).is_err());
        }
    }

    #[test]
    fn absent_endpoint_fields_are_rejected_but_explicit_null_is_allowed() {
        let valid = serde_json::to_value(segment("hold", None, None, None)).unwrap();
        assert!(serde_json::from_value::<SessionResult>(valid.clone()).is_ok());
        for field in ["reference_end_ms", "baseline_end_ms", "candidate_end_ms"] {
            let mut missing = valid.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<SessionResult>(missing).is_err());
        }
    }

    #[test]
    fn subtracts_integer_timestamps_before_float_conversion() {
        let report = evaluate(&[segment(
            "large-timestamp",
            Some(u64::MAX - 1),
            Some(u64::MAX),
            Some(u64::MAX - 1),
        )])
        .unwrap();
        assert_eq!(report.baseline.endpoint_delay_ms.p50, Some(1.0));
        assert_eq!(report.candidate.endpoint_delay_ms.p50, Some(0.0));
    }
}
