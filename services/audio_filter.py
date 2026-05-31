"""Manual audio preprocessing filters using only numpy (no scipy)."""

from __future__ import annotations

import numpy as np


# -------------------------
# Gain
# -------------------------
def apply_gain(samples: np.ndarray, gain: float) -> np.ndarray:
    """Apply microphone gain and clip samples to -1..1 range."""
    audio = np.asarray(samples, dtype=np.float32)
    return np.clip(audio * float(gain), -1.0, 1.0).astype(np.float32, copy=False)


# -------------------------
# Simple High-Pass Filter
# (1st-order difference filter approximation)
# -------------------------
def high_pass_filter(
    samples: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
) -> np.ndarray:
    """
    Lightweight high-pass filter using a simple RC-like difference method.
    No scipy required.
    """
    audio = np.asarray(samples, dtype=np.float32)

    if audio.size == 0 or sample_rate <= 0 or cutoff_hz <= 0:
        return audio.astype(np.float32, copy=False)

    # smoothing factor (RC approximation)
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = rc / (rc + dt)

    filtered = np.zeros_like(audio, dtype=np.float32)
    prev = 0.0

    for i in range(audio.size):
        prev = alpha * (prev + audio[i] - (audio[i - 1] if i > 0 else 0.0))
        filtered[i] = prev

    return filtered.astype(np.float32, copy=False)


# -------------------------
# Simple Noise Reduction (FFT spectral gating)
# -------------------------
def reduce_noise(
    samples: np.ndarray,
    sample_rate: int,
    prop_decrease: float = 0.8,
    stationary: bool = True,
) -> np.ndarray:
    """
    Simple FFT-based noise reduction (spectral subtraction approximation).
    """
    audio = np.asarray(samples, dtype=np.float32)
    n = audio.size

    if n == 0:
        return audio.astype(np.float32, copy=False)

    strength = float(np.clip(prop_decrease, 0.1, 1.0))

    # FFT
    spectrum = np.fft.rfft(audio)
    magnitude = np.abs(spectrum)
    phase = np.exp(1j * np.angle(spectrum))

    # Estimate noise from first 0.5s
    noise_samples = min(int(0.5 * sample_rate), n)
    noise_spectrum = np.fft.rfft(audio[:noise_samples])
    noise_profile = np.abs(noise_spectrum)

    if not stationary:
        # crude alternative: low percentile energy floor
        noise_profile = np.minimum(
            noise_profile,
            np.percentile(magnitude, 10, axis=0, keepdims=False)
        )

    # spectral subtraction
    floor = (1.0 - strength) * magnitude
    cleaned = np.maximum(magnitude - strength * noise_profile, floor)

    reconstructed = np.fft.irfft(cleaned * phase, n=n)

    reconstructed = np.clip(reconstructed, -1.0, 1.0).astype(np.float32)

    return reconstructed


# -------------------------
# Full Pipeline
# -------------------------
def preprocess(
    samples: np.ndarray,
    sample_rate: int = 16_000,
    *,
    gain: float = 1.0,
    denoise: bool = True,
    highpass: bool = True,
    highpass_cutoff_hz: float = 80.0,
    denoise_prop_decrease: float = 0.8,
    denoise_stationary: bool = True,
) -> np.ndarray:
    """Run gain, high-pass filtering, and noise reduction."""

    audio = apply_gain(np.asarray(samples, dtype=np.float32), gain)

    original_length = audio.size

    if highpass:
        audio = high_pass_filter(audio, sample_rate, highpass_cutoff_hz)

    if denoise:
        audio = reduce_noise(
            audio,
            sample_rate,
            prop_decrease=denoise_prop_decrease,
            stationary=denoise_stationary,
        )

    # restore length
    if audio.size < original_length:
        audio = np.pad(audio, (0, original_length - audio.size))
    elif audio.size > original_length:
        audio = audio[:original_length]

    return audio.astype(np.float32)


__all__ = [
    "apply_gain",
    "high_pass_filter",
    "reduce_noise",
    "preprocess",
]
