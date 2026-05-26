"""Manual audio preprocessing filters using only numpy and scipy."""

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
    """Apply a fourth-order high-pass filter to remove low-frequency rumble."""

    try:
        import numpy as np
        from scipy.signal import butter, sosfilt
    except ImportError:
        return samples

    audio = np.asarray(samples, dtype=np.float32)
    if audio.size == 0 or sample_rate <= 0 or cutoff_hz <= 0:
        return audio.astype(np.float32, copy=False)

    nyquist = sample_rate / 2.0
    cutoff = min(float(cutoff_hz), nyquist * 0.95)
    if cutoff <= 0:
        return audio.astype(np.float32, copy=False)

    sos = butter(4, cutoff, btype="high", fs=sample_rate, output="sos")
    filtered = sosfilt(sos, audio)
    return np.asarray(filtered, dtype=np.float32)


def reduce_noise(
    samples: "np.ndarray",
    sample_rate: int,
    prop_decrease: float = 0.8,
    stationary: bool = True,
) -> "np.ndarray":
    """Reduce stationary background noise with simple spectral subtraction."""

    try:
        import numpy as np
        from scipy.signal import istft, stft
    except ImportError:
        return samples

    audio = np.asarray(samples, dtype=np.float32)
    original_length = int(audio.size)
    if original_length == 0:
        return audio.astype(np.float32, copy=False)

    strength = float(np.clip(prop_decrease, 0.1, 1.0))
    _, _, spectrum = stft(audio, fs=sample_rate, nperseg=512)
    magnitude = np.abs(spectrum)
    phase = np.exp(1j * np.angle(spectrum))
    if magnitude.size == 0:
        return audio.astype(np.float32, copy=False)

    noise_seconds = max(1, int(round(0.5 * sample_rate)))
    _, _, noise_spectrum = stft(audio[:noise_seconds], fs=sample_rate, nperseg=512)
    noise_profile = np.mean(np.abs(noise_spectrum), axis=1, keepdims=True)
    if not stationary:
        quiet_profile = np.percentile(magnitude, 10, axis=1, keepdims=True)
        noise_profile = np.minimum(noise_profile, quiet_profile)

    floor = (1.0 - strength) * magnitude
    cleaned_magnitude = np.maximum(magnitude - (strength * noise_profile), floor)
    _, reconstructed = istft(cleaned_magnitude * phase, fs=sample_rate, nperseg=512)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    if reconstructed.size < original_length:
        reconstructed = np.pad(reconstructed, (0, original_length - reconstructed.size))
    elif reconstructed.size > original_length:
        reconstructed = reconstructed[:original_length]
    return np.clip(reconstructed, -1.0, 1.0).astype(np.float32, copy=False)


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
