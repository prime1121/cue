//! Rust translation of the Pipecat/Transformers Whisper frontend, modified by SAO Labs.
//!
//! Copyright (c) 2024-2026 Daily (BSD-2-Clause).
//! Portions copyright 2022 The HuggingFace Inc. team; copyright 2023 The HuggingFace
//! Inc. team and the librosa & torchaudio authors (Apache-2.0).
//! Sources and retained license texts: `vendor/PROVENANCE.md`.

use anyhow::{ensure, Result};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::{Arc, OnceLock};

pub const SAMPLE_RATE: usize = 16_000;
pub const CONTEXT_SAMPLES: usize = SAMPLE_RATE * 8;
pub const MEL_BINS: usize = 80;
pub const MEL_FRAMES: usize = 800;
const FFT_SIZE: usize = 400;
const HOP_SIZE: usize = 160;
const POWER_BINS: usize = FFT_SIZE / 2 + 1;

struct Frontend {
    fft: Arc<dyn Fft<f64>>,
    window: [f64; FFT_SIZE],
    filters: Vec<Vec<(usize, f64)>>,
}

impl Frontend {
    fn new() -> Self {
        let fft = FftPlanner::<f64>::new().plan_fft_forward(FFT_SIZE);
        let window = std::array::from_fn(|i| {
            0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / FFT_SIZE as f64).cos()
        });
        let max_mel = 15.0 + (8.0_f64).ln() * 27.0 / 6.4_f64.ln();
        let frequencies: Vec<f64> = (0..MEL_BINS + 2)
            .map(|i| {
                let mel = max_mel * i as f64 / (MEL_BINS + 1) as f64;
                if mel < 15.0 {
                    200.0 * mel / 3.0
                } else {
                    1000.0 * ((mel - 15.0) * 6.4_f64.ln() / 27.0).exp()
                }
            })
            .collect();
        let filters = (0..MEL_BINS)
            .map(|m| {
                let left = frequencies[m];
                let center = frequencies[m + 1];
                let right = frequencies[m + 2];
                (0..POWER_BINS)
                    .filter_map(|bin| {
                        let hz = bin as f64 * SAMPLE_RATE as f64 / FFT_SIZE as f64;
                        let weight = ((hz - left) / (center - left))
                            .min((right - hz) / (right - center))
                            .max(0.0)
                            * 2.0
                            / (right - left);
                        (weight > 0.0).then_some((bin, weight))
                    })
                    .collect()
            })
            .collect();
        Self {
            fft,
            window,
            filters,
        }
    }
}

/// Convert mono 16 kHz float PCM into row-major `[80, 800]` log-mel features.
///
/// Keeps the last eight seconds and pads shorter input on the **left**, before
/// normalizing the entire 128,000-sample window. Empty input means silence.
/// Non-finite samples are rejected rather than poisoning the model state.
pub fn log_mel(audio_16k: &[f32]) -> Result<Vec<f32>> {
    let audio = &audio_16k[audio_16k.len().saturating_sub(CONTEXT_SAMPLES)..];
    ensure!(
        audio.iter().all(|x| x.is_finite()),
        "audio contains NaN or infinity"
    );
    let mut waveform = vec![0.0_f32; CONTEXT_SAMPLES];
    waveform[CONTEXT_SAMPLES - audio.len()..].copy_from_slice(audio);

    // NumPy normalizes in float32 before its FFT promotes to float64. Accumulate
    // sums in f64 to avoid length-dependent error from a serial f32 reduction.
    let mean =
        (waveform.iter().map(|&x| f64::from(x)).sum::<f64>() / CONTEXT_SAMPLES as f64) as f32;
    let variance = (waveform
        .iter()
        .map(|&x| {
            let d = x - mean;
            f64::from(d * d)
        })
        .sum::<f64>()
        / CONTEXT_SAMPLES as f64) as f32;
    ensure!(
        mean.is_finite() && variance.is_finite(),
        "audio exceeds supported float PCM range"
    );
    let divisor = (variance + 1e-7_f32).sqrt();
    for x in &mut waveform {
        *x = (*x - mean) / divisor;
    }

    static FRONTEND: OnceLock<Frontend> = OnceLock::new();
    let frontend = FRONTEND.get_or_init(Frontend::new);
    let mut spectrum = [Complex::new(0.0, 0.0); FFT_SIZE];
    let mut scratch = vec![Complex::new(0.0, 0.0); frontend.fft.get_inplace_scratch_len()];
    let mut power = [0.0_f64; POWER_BINS];
    let mut features = vec![0.0_f64; MEL_BINS * MEL_FRAMES];
    let mut maximum = f64::NEG_INFINITY;

    // Centered STFT with reflect padding. The reference's 801st frame is dropped
    // before dynamic-range clipping, so it need not be computed here.
    for frame in 0..MEL_FRAMES {
        for (i, value) in spectrum.iter_mut().enumerate() {
            let offset = (frame * HOP_SIZE + i) as isize - (FFT_SIZE / 2) as isize;
            let source = if offset < 0 {
                (-offset) as usize
            } else if offset >= CONTEXT_SAMPLES as isize {
                (2 * CONTEXT_SAMPLES as isize - 2 - offset) as usize
            } else {
                offset as usize
            };
            *value = Complex::new(f64::from(waveform[source]) * frontend.window[i], 0.0);
        }
        frontend
            .fft
            .process_with_scratch(&mut spectrum, &mut scratch);
        for (bin, value) in power.iter_mut().enumerate() {
            *value = spectrum[bin].norm_sqr();
        }
        for (mel, filter) in frontend.filters.iter().enumerate() {
            let energy = filter
                .iter()
                .map(|&(bin, weight)| power[bin] * weight)
                .sum::<f64>();
            let value = energy.max(1e-10).log10();
            features[mel * MEL_FRAMES + frame] = value;
            maximum = maximum.max(value);
        }
    }
    Ok(features
        .into_iter()
        .map(|x| ((x.max(maximum - 8.0) + 4.0) / 4.0) as f32)
        .collect())
}
