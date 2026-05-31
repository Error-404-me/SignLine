"""Manual audio preprocessing filters using only numpy — Android safe."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def apply_gain(samples: "np.ndarray", gain: float) -> "np.ndarray":
    """Apply microphone gain and clip samples to the -1..1 range."""

    try:
        import numpy as np
    except ImportError:
        return samples

    audio = np.asarray(samples, dtype=np.float32)
    return np.clip(audio * float(gain), -1.0, 1.0).astype(np.float32, copy=False)


def high_pass_filter(
    samples: "np.ndarray",
    sample_rate: int,
    cutoff_hz: float,
) -> "np.ndarray":
    """First-order high-pass filter using numpy only — removes low-frequency rumble."""

    try:
        import numpy as np
    except ImportError:
        return samples

    audio = np.asarray(samples, dtype=np.float32)
    if audio.size == 0 or sample_rate <= 0 or cutoff_hz <= 0:
        return audio.astype(np.float32, copy=False)

    # RC high-pass filter: y[i] = alpha * (y[i-1] + x[i] - x[i-1])
    dt = 1.0 / float(sample_rate)
    rc = 1.0 / (2.0 * np.pi * float(cutoff_hz))
    alpha = rc / (rc + dt)

    result = np.empty_like(audio)
    result[0] = audio[0]
    for i in range(1, len(audio)):
        result[i] = alpha * (result[i - 1] + audio[i] - audio[i - 1])

    return result.astype(np.float32, copy=False)


def reduce_noise(
    samples: "np.ndarray",
    sample_rate: int,
    prop_decrease: float = 0.8,
    stationary: bool = True,
) -> "np.ndarray":
    """Spectral subtraction noise reduction using numpy FFT only."""

    try:
        import numpy as np
    except ImportError:
        return samples

    audio = np.asarray(samples, dtype=np.float32)
    original_length = int(audio.size)
    if original_length == 0:
        return audio.astype(np.float32, copy=False)

    strength = float(np.clip(prop_decrease, 0.1, 1.0))
    n_fft = 512
    hop = n_fft // 2
    window = np.hanning(n_fft)

    # Estimate noise profile from first 0.5s
    noise_end = min(int(0.5 * sample_rate), original_length)
    noise_slice = audio[:noise_end]
    if len(noise_slice) < n_fft:
        noise_slice = np.pad(noise_slice, (0, n_fft - len(noise_slice)))

    noise_frames = []
    for i in range(0, len(noise_slice) - n_fft + 1, hop):
        frame = noise_slice[i:i + n_fft] * window
        noise_frames.append(np.abs(np.fft.rfft(frame)))
    noise_profile = (
        np.mean(noise_frames, axis=0)
        if noise_frames
        else np.zeros(n_fft // 2 + 1, dtype=np.float32)
    )

    if not stationary:
        # For variable noise: use quieter floor estimate
        all_frames = []
        for i in range(0, original_length - n_fft + 1, hop):
            frame = audio[i:i + n_fft] * window
            all_frames.append(np.abs(np.fft.rfft(frame)))
        if all_frames:
            all_mag = np.stack(all_frames, axis=1)
            quiet_profile = np.percentile(all_mag, 10, axis=1)
            noise_profile = np.minimum(noise_profile, quiet_profile)

    # Pad signal to fit whole frames
    pad_length = (n_fft - original_length % n_fft) % n_fft
    padded = np.pad(audio, (0, pad_length + n_fft))
    output = np.zeros(len(padded), dtype=np.float32)
    counts = np.zeros(len(padded), dtype=np.float32)

    for i in range(0, original_length, hop):
        frame = padded[i:i + n_fft]
        if len(frame) < n_fft:
            break
        windowed = frame * window
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Spectral subtraction
        floor = (1.0 - strength) * magnitude
        cleaned_mag = np.maximum(magnitude - strength * noise_profile, floor)
        cleaned = cleaned_mag * np.exp(1j * phase)
        reconstructed = np.fft.irfft(cleaned) * window

        output[i:i + n_fft] += reconstructed
        counts[i:i + n_fft] += window ** 2

    # Normalize overlap-add
    nonzero = counts > 1e-8
    output[nonzero] /= counts[nonzero]

    result = output[:original_length].astype(np.float32)
    return np.clip(result, -1.0, 1.0).astype(np.float32, copy=False)


def preprocess(
    samples: "np.ndarray",
    sample_rate: int = 16_000,
    *,
    gain: float = 1.0,
    denoise: bool = True,
    highpass: bool = True,
    highpass_cutoff_hz: float = 80.0,
    denoise_prop_decrease: float = 0.8,
    denoise_stationary: bool = True,
) -> "np.ndarray":
    """Run gain, high-pass filtering, and spectral denoise in order."""

    try:
        import numpy as np
    except ImportError:
        return samples

    original_length = int(len(samples))
    audio = apply_gain(np.asarray(samples, dtype=np.float32), gain)
    if highpass:
        audio = high_pass_filter(audio, sample_rate, highpass_cutoff_hz)
    if denoise:
        audio = reduce_noise(
            audio,
            sample_rate,
            prop_decrease=denoise_prop_decrease,
            stationary=denoise_stationary,
        )
    if audio.size < original_length:
        audio = np.pad(audio, (0, original_length - audio.size))
    elif audio.size > original_length:
        audio = audio[:original_length]
    return np.asarray(audio, dtype=np.float32)


__all__ = [
    "apply_gain",
    "high_pass_filter",
    "preprocess",
    "reduce_noise",
]
