"""Offline faster-whisper speech-to-text backend.

This module is intentionally capture-agnostic. It does not open a microphone,
call cloud APIs, or depend on desktop-only audio packages. The Flet UI layer is
expected to provide WAV bytes or raw PCM bytes from whatever capture mechanism
is available on the current platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import os
from pathlib import Path
import threading
import time
import wave
from typing import Any, Mapping

import numpy as np


WHISPER_SAMPLE_RATE = 16_000
DEFAULT_VAD_PARAMETERS: dict[str, int | float] = {
    "min_silence_duration_ms": 500,
    "threshold": 0.5,
}


@dataclass(frozen=True, slots=True)
class ASRConfig:
    """Configuration for local faster-whisper inference."""

    model_size_or_path: str | Path = "base"
    language: str | None = None
    device: str = "auto"
    compute_type: str = "int8"
    cpu_threads: int = 0
    num_workers: int = 1
    download_root: Path | None = None
    local_files_only: bool = False
    beam_size: int = 1
    vad_filter: bool = True
    vad_parameters: Mapping[str, Any] | None = field(
        default_factory=lambda: dict(DEFAULT_VAD_PARAMETERS)
    )
    mic_gain: float = 1.0
    denoise_enabled: bool = True
    highpass_enabled: bool = True
    highpass_cutoff_hz: float = 80.0
    denoise_prop_decrease: float = 0.8
    denoise_stationary: bool = True
    raw_sample_rate: int = WHISPER_SAMPLE_RATE
    raw_channels: int = 1
    raw_sample_width: int = 2
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.6
    vad_silence_ms: int = 500
    vad_threshold: float = 0.5
    initial_prompt: str | None = None


class ASRService:
    """Load a local Whisper model once and transcribe WAV or PCM bytes."""

    def __init__(self, config: ASRConfig | None = None) -> None:
        self.config = config or ASRConfig()
        self._model: Any | None = None
        self._model_lock = threading.RLock()
        self._transcribe_lock = threading.RLock()

    def transcribe_bytes(self, audio_bytes: bytes) -> dict[str, str | float]:
        """Transcribe one WAV or raw PCM byte payload.

        WAV bytes are decoded with the standard-library ``wave`` module. Raw
        PCM bytes are interpreted from ``ASRConfig.raw_*`` fields and converted
        without using deprecated standard-library audio helpers.
        """

        waveform, duration = self._decode_audio_bytes(audio_bytes)
        return self._transcribe_waveform(waveform, duration)

    def _load_model(self) -> Any:
        """Create the faster-whisper model lazily and reuse it."""

        with self._model_lock:
            if self._model is not None:
                return self._model

            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "Install faster-whisper to use offline speech recognition."
                ) from exc

            download_root = (
                str(self.config.download_root)
                if self.config.download_root is not None
                else None
            )
            self._model = WhisperModel(
                str(self.config.model_size_or_path),
                device=self.config.device,
                compute_type=self.config.compute_type,
                cpu_threads=self.config.cpu_threads,
                num_workers=self.config.num_workers,
                download_root=download_root,
                local_files_only=self.config.local_files_only,
            )
            return self._model

    def _transcribe_waveform(
        self,
        waveform: np.ndarray,
        duration: float,
    ) -> dict[str, str | float]:
        """Run faster-whisper on a normalized 16 kHz mono waveform."""

        if waveform.size == 0:
            return {
                "text": "",
                "language": self.config.language or "",
                "duration": 0.0,
            }

        model = self._load_model()
        vad_parameters = {
            "min_silence_duration_ms": self.config.vad_silence_ms,
            "threshold": self.config.vad_threshold,
        }
        with self._transcribe_lock:
            segments, info = model.transcribe(
                waveform,
                language=self.config.language,
                task="transcribe",
                beam_size=self.config.beam_size,
                vad_filter=self.config.vad_filter,
                vad_parameters=vad_parameters,
                condition_on_previous_text=self.config.condition_on_previous_text,
                initial_prompt=self.config.initial_prompt,
                temperature=0.0,
                no_speech_threshold=self.config.no_speech_threshold,
                without_timestamps=True,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()

        return {
            "text": _compact_spaces(text),
            "language": getattr(info, "language", None) or self.config.language or "",
            "duration": round(float(duration), 3),
        }

    def _decode_audio_bytes(self, audio_bytes: bytes) -> tuple[np.ndarray, float]:
        """Decode WAV bytes or raw PCM bytes to 16 kHz mono float32 samples."""

        if not audio_bytes:
            return np.empty(0, dtype=np.float32), 0.0

        if _looks_like_wav(audio_bytes):
            samples, source_rate = _decode_wav_bytes(audio_bytes)
        else:
            samples = _pcm_to_float32(
                audio_bytes,
                sample_width=self.config.raw_sample_width,
                channels=self.config.raw_channels,
            )
            source_rate = self.config.raw_sample_rate

        if source_rate != WHISPER_SAMPLE_RATE:
            samples = _resample_linear(samples, source_rate, WHISPER_SAMPLE_RATE)

        samples = np.ascontiguousarray(samples, dtype=np.float32)
        from services.audio_filter import preprocess

        samples = preprocess(
            samples,
            WHISPER_SAMPLE_RATE,
            gain=self.config.mic_gain,
            denoise=self.config.denoise_enabled,
            highpass=self.config.highpass_enabled,
            highpass_cutoff_hz=self.config.highpass_cutoff_hz,
            denoise_prop_decrease=self.config.denoise_prop_decrease,
            denoise_stationary=self.config.denoise_stationary,
        )
        duration = samples.size / WHISPER_SAMPLE_RATE
        return samples, duration


class RealtimeASRBuffer:
    """Chunk buffer for near real-time captions.

    Incoming audio chunks are normalized into 16 kHz mono samples, held in
    memory, then transcribed once 1-2 seconds of audio is available. The buffer
    is cleared for each emitted partial result.
    """

    def __init__(
        self,
        asr_service: ASRService,
        *,
        min_buffer_seconds: float = 1.0,
        max_buffer_seconds: float = 2.0,
        min_process_interval_seconds: float = 0.75,
    ) -> None:
        if min_buffer_seconds <= 0:
            raise ValueError("min_buffer_seconds must be greater than zero.")
        if max_buffer_seconds < min_buffer_seconds:
            raise ValueError("max_buffer_seconds must be >= min_buffer_seconds.")

        self.asr_service = asr_service
        self.min_buffer_seconds = min_buffer_seconds
        self.max_buffer_seconds = max_buffer_seconds
        self.min_process_interval_seconds = min_process_interval_seconds
        self._chunks: list[np.ndarray] = []
        self._sample_count = 0
        self._last_process_at = time.monotonic()
        self._lock = threading.RLock()

    @property
    def buffered_seconds(self) -> float:
        """Return the amount of queued audio."""

        with self._lock:
            return self._sample_count / WHISPER_SAMPLE_RATE

    def clear(self) -> None:
        """Discard pending audio chunks."""

        with self._lock:
            self._chunks.clear()
            self._sample_count = 0

    def transcribe_chunk(self, audio_bytes: bytes) -> str:
        """Accept one audio chunk and return text only when a partial is ready."""

        waveform, _ = self.asr_service._decode_audio_bytes(audio_bytes)
        if waveform.size == 0:
            return ""

        with self._lock:
            self._chunks.append(waveform)
            self._sample_count += waveform.size
            buffered_seconds = self._sample_count / WHISPER_SAMPLE_RATE
            elapsed = time.monotonic() - self._last_process_at
            should_process = buffered_seconds >= self.max_buffer_seconds or (
                buffered_seconds >= self.min_buffer_seconds
                and elapsed >= self.min_process_interval_seconds
            )
            if not should_process:
                return ""
            audio = self._take_buffer_locked()

        return self._transcribe_buffer(audio)

    def flush(self) -> str:
        """Transcribe and clear any remaining buffered audio."""

        with self._lock:
            if self._sample_count == 0:
                return ""
            audio = self._take_buffer_locked()
        return self._transcribe_buffer(audio)

    def _take_buffer_locked(self) -> np.ndarray:
        """Return concatenated buffered audio and reset buffer state."""

        audio = np.concatenate(self._chunks).astype(np.float32, copy=False)
        self._chunks.clear()
        self._sample_count = 0
        self._last_process_at = time.monotonic()
        return audio

    def _transcribe_buffer(self, audio: np.ndarray) -> str:
        duration = audio.size / WHISPER_SAMPLE_RATE
        result = self.asr_service._transcribe_waveform(audio, duration)
        return str(result["text"]).strip()


_DEFAULT_SERVICE: ASRService | None = None
_DEFAULT_BUFFER: RealtimeASRBuffer | None = None
_DEFAULT_LOCK = threading.RLock()


def transcribe_bytes(audio_bytes: bytes) -> dict[str, str | float]:
    """Backend API: transcribe a complete WAV or raw PCM byte payload."""

    return _default_service().transcribe_bytes(audio_bytes)


def transcribe_chunk(audio_bytes: bytes) -> str:
    """Backend API: feed one live audio chunk and receive incremental text."""

    return _default_buffer().transcribe_chunk(audio_bytes)


def _default_service() -> ASRService:
    global _DEFAULT_SERVICE

    with _DEFAULT_LOCK:
        if _DEFAULT_SERVICE is None:
            _DEFAULT_SERVICE = ASRService(_config_from_environment())
        return _DEFAULT_SERVICE


def _default_buffer() -> RealtimeASRBuffer:
    global _DEFAULT_BUFFER

    with _DEFAULT_LOCK:
        if _DEFAULT_BUFFER is None:
            _DEFAULT_BUFFER = RealtimeASRBuffer(_default_service())
        return _DEFAULT_BUFFER


def _config_from_environment() -> ASRConfig:
    language = os.getenv("SPEAK_SIGN_ASR_LANGUAGE") or None
    model = os.getenv("SPEAK_SIGN_ASR_MODEL", "base")

    # Resolve assets/models/ relative to this file:
    # asr_service.py → services/ → speak_sign_app/ → assets/models/
    _services_dir = Path(__file__).parent
    _assets_models_dir = _services_dir.parent / "assets" / "models"

    local_model_dir = _assets_models_dir / model
    if local_model_dir.exists():
        # Use bundled CTranslate2 model folder directly
        model_size_or_path = local_model_dir
        local_files_only = True
    else:
        # Fall back to HuggingFace download/cache
        model_size_or_path = model
        local_files_only = False

    download_root_str = os.getenv("SPEAK_SIGN_ASR_DOWNLOAD_ROOT")

    return ASRConfig(
        model_size_or_path=model_size_or_path,
        language=language,
        download_root=Path(download_root_str) if download_root_str else None,
        local_files_only=local_files_only,
    )


def _looks_like_wav(audio_bytes: bytes) -> bool:
    return audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"


def _decode_wav_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        pcm_bytes = wav_file.readframes(frame_count)

    samples = _pcm_to_float32(
        pcm_bytes,
        sample_width=sample_width,
        channels=channels,
    )
    return samples, sample_rate


def _pcm_to_float32(
    pcm_bytes: bytes,
    *,
    sample_width: int,
    channels: int,
) -> np.ndarray:
    if channels <= 0:
        raise ValueError("Audio channel count must be greater than zero.")
    if sample_width not in {1, 2, 3, 4}:
        raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes.")
    if len(pcm_bytes) == 0:
        return np.empty(0, dtype=np.float32)

    frame_width = sample_width * channels
    if len(pcm_bytes) % frame_width:
        usable_bytes = len(pcm_bytes) - (len(pcm_bytes) % frame_width)
        pcm_bytes = pcm_bytes[:usable_bytes]

    if sample_width == 1:
        samples = (
            np.frombuffer(pcm_bytes, dtype=np.uint8).astype(np.float32) - 128.0
        ) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        samples = _pcm24_to_float32(pcm_bytes)
    else:
        samples = (
            np.frombuffer(pcm_bytes, dtype="<i4").astype(np.float32) / 2147483648.0
        )

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    return np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)


def _pcm24_to_float32(pcm_bytes: bytes) -> np.ndarray:
    raw = np.frombuffer(pcm_bytes, dtype=np.uint8).reshape(-1, 3)
    values = (
        raw[:, 0].astype(np.int32)
        | (raw[:, 1].astype(np.int32) << 8)
        | (raw[:, 2].astype(np.int32) << 16)
    )
    values = np.where(values & 0x800000, values | ~0xFFFFFF, values)
    return values.astype(np.float32) / 8388608.0


def _resample_linear(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    if source_rate <= 0:
        raise ValueError("Audio sample rate must be greater than zero.")
    if samples.size == 0 or source_rate == target_rate:
        return samples.astype(np.float32, copy=False)

    duration = samples.size / source_rate
    target_size = max(1, int(round(duration * target_rate)))
    source_positions = np.linspace(0.0, samples.size - 1, num=samples.size)
    target_positions = np.linspace(0.0, samples.size - 1, num=target_size)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _compact_spaces(text: str) -> str:
    return " ".join(text.split())


if __name__ == "__main__":
    # Dummy streaming example. Replace the silence bytes with chunks emitted by
    # the Flet audio capture layer.
    one_quarter_second_of_silence = b"\x00\x00" * (WHISPER_SAMPLE_RATE // 4)
    service = ASRService(ASRConfig(model_size_or_path="tiny"))
    stream = RealtimeASRBuffer(service)

    for _ in range(8):
        partial_text = stream.transcribe_chunk(one_quarter_second_of_silence)
        if partial_text:
            print(f"partial: {partial_text}")

    final_text = stream.flush()
    if final_text:
        print(f"final: {final_text}")


__all__ = [
    "ASRConfig",
    "ASRService",
    "RealtimeASRBuffer",
    "transcribe_bytes",
    "transcribe_chunk",
]
