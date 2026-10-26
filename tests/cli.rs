use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new(body: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "sao-turn-test-{}-{}.jsonl",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, body).unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[test]
fn replay_obeys_number_collection_context() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/policy-demo.jsonl");
    let output = Command::new(env!("CARGO_BIN_EXE_sao-turn"))
        .arg("replay")
        .arg(path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let rows: Vec<serde_json::Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows[0]["decision"]["kind"], "wait");
    assert_eq!(rows[0]["baseline_decision"]["kind"], "end");
    assert_eq!(rows[0]["baseline_decision"]["at_ms"], 500);
    assert_eq!(rows.last().unwrap()["decision"]["kind"], "end");
    assert_eq!(rows.last().unwrap()["decision"]["at_ms"], 1400);
}

#[test]
fn benchmark_requires_explicit_ground_truth_before_loading_models() {
    let file = Fixture::new("{\"id\":\"unlabelled\",\"audio\":\"missing.wav\"}\n");
    let output = Command::new(env!("CARGO_BIN_EXE_sao-turn"))
        .arg("benchmark")
        .arg(&file.0)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("reference_end_ms"));
}

#[test]
fn compare_rejects_absent_prediction_fields() {
    let file = Fixture::new("{\"id\":\"invalid\",\"reference_end_ms\":null,\"baseline_end_ms\":null,\"baseline_inference_ms\":[],\"candidate_inference_ms\":[]}\n");
    let output = Command::new(env!("CARGO_BIN_EXE_sao-turn"))
        .arg("compare")
        .arg(&file.0)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("candidate_end_ms"));
}
