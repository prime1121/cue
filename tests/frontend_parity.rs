use sao_turn::features::{log_mel, CONTEXT_SAMPLES, MEL_BINS, MEL_FRAMES};

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn probe() -> Vec<f32> {
    floats(include_bytes!("../assets/frontend/probe.f32"))
}

fn assert_parity(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), MEL_BINS * MEL_FRAMES);
    assert_eq!(actual.len(), expected.len());
    let (worst, maximum) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (a, b))| (index, (a - b).abs()))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    let mean = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / actual.len() as f64;
    eprintln!("frontend parity: max_abs={maximum:.9}, mean_abs={mean:.9}, worst_index={worst}");
    assert!(
        maximum <= tolerance,
        "frontend mismatch at {worst}: {} versus {}, max error {maximum}",
        actual[worst],
        expected[worst]
    );
}

#[test]
fn complete_window_matches_frozen_pipecat_reference() {
    let input = probe();
    let expected = floats(include_bytes!("../assets/frontend/long.mel.f32"));
    assert_parity(&log_mel(&input).unwrap(), &expected, 3e-5);
}

#[test]
fn left_padded_short_audio_matches_frozen_pipecat_reference() {
    let input = probe();
    let expected = floats(include_bytes!("../assets/frontend/short.mel.f32"));
    assert_parity(
        &log_mel(&input[input.len() - 20513..]).unwrap(),
        &expected,
        3e-5,
    );
}

#[test]
fn silence_empty_and_constant_full_window_are_finite() {
    for audio in [Vec::new(), vec![0.0; 17], vec![0.5; CONTEXT_SAMPLES]] {
        let features = log_mel(&audio).unwrap();
        assert_eq!(features.len(), MEL_BINS * MEL_FRAMES);
        assert!(features.iter().all(|value| *value == -1.5));
    }
}

#[test]
fn frontend_keeps_the_trailing_eight_seconds_and_rejects_nonfinite_audio() {
    let input = probe();
    assert_eq!(
        log_mel(&input).unwrap(),
        log_mel(&input[input.len() - CONTEXT_SAMPLES..]).unwrap()
    );
    assert!(log_mel(&[f32::NAN]).is_err());
    assert!(log_mel(&[f32::INFINITY]).is_err());
}

#[test]
#[ignore = "requires checksum-verified model files and ORT_DYLIB_PATH; run cargo test --test frontend_parity -- --ignored"]
fn native_models_match_python_onnxruntime_and_vad_reset() {
    use sao_turn::inference::{SileroVad, SmartTurnModel};
    use std::path::Path;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let runtime = std::env::var_os("ORT_DYLIB_PATH").map(std::path::PathBuf::from);
    let model_dir = root.join("assets/models");
    let mut smart = SmartTurnModel::load(
        &model_dir.join("smart-turn-v3.2-cpu.onnx"),
        runtime.as_deref(),
        1,
    )
    .unwrap();
    let mut vad =
        SileroVad::load(&model_dir.join("silero-vad.onnx"), runtime.as_deref(), 1).unwrap();
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("../assets/frontend/reference.json")).unwrap();
    let input = probe();
    for (name, samples) in [
        ("long", &input[..]),
        ("short", &input[input.len() - 20513..]),
        ("silence", &[0.0_f32; 512][..]),
    ] {
        let expected = reference["smart_turn_probabilities"][name]
            .as_f64()
            .unwrap() as f32;
        let started = std::time::Instant::now();
        let actual = smart.predict(samples).unwrap();
        eprintln!(
            "Smart Turn {name}: native={actual:.9}, reference={expected:.9}, elapsed_ms={:.3}",
            started.elapsed().as_secs_f64() * 1000.0
        );
        // Dynamic quantization can amplify tiny FFT rounding changes.
        assert!(
            (actual - expected).abs() <= 3e-3,
            "{name}: {actual} != {expected}"
        );
    }
    assert!(smart.predict(&[]).is_err());
    assert!(vad.process(&[0.0; 511]).is_err());
    assert!(vad.process(&[f32::NAN; 512]).is_err());
    let started = std::time::Instant::now();
    for (frame, expected) in input
        .chunks_exact(512)
        .zip(reference["silero_probabilities"].as_array().unwrap())
    {
        assert!((vad.process(frame).unwrap() - expected.as_f64().unwrap() as f32).abs() <= 2e-5);
    }
    eprintln!(
        "Silero 64 frames: total_ms={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );
    vad.reset();
    let fresh = vad.process(&input[..512]).unwrap();
    let expected = reference["silero_probabilities"][0].as_f64().unwrap() as f32;
    assert!((fresh - expected).abs() <= 2e-5);
}
