"""Offline Vosk speech-to-text backend for live PCM chunks.

This module is capture-agnostic. It does not open a microphone or call online
APIs. The UI layer provides 16 kHz mono int16 PCM chunks or complete WAV bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import threading
import wave
from typing import Any

import numpy as np


VOSK_SAMPLE_RATE = 16_000
PCM_SAMPLE_WIDTH = 2


@dataclass(frozen=True, slots=True)
class VoskASRConfig:
    """Configuration for local Vosk inference."""

    model_path: Path
    language: str = "en"
    sample_rate: int = VOSK_SAMPLE_RATE
    return_partials: bool = False


class VoskASRService:
    """Load one local Vosk model and transcribe WAV or PCM bytes."""

    def __init__(self, config: VoskASRConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._model_lock = threading.RLock()

    def transcribe_bytes(self, audio_bytes: bytes) -> dict[str, str | float]:
        """Transcribe one complete WAV or 16 kHz mono int16 PCM payload."""

        pcm_bytes, duration = self._decode_audio_bytes(audio_bytes)
        if not pcm_bytes:
            return {
                "text": "",
                "language": self.config.language,
                "duration": 0.0,
            }
        recognizer = self.create_recognizer()
        texts: list[str] = []
        one_second = self.config.sample_rate * PCM_SAMPLE_WIDTH
        for offset in range(0, len(pcm_bytes), one_second):
            chunk = pcm_bytes[offset : offset + one_second]
            if recognizer.AcceptWaveform(chunk):
                texts.append(_text_from_result(recognizer.Result()))
        texts.append(_text_from_result(recognizer.FinalResult()))
        return {
            "text": _compact_spaces(" ".join(texts)),
            "language": self.config.language,
            "duration": round(duration, 3),
        }

    def create_recognizer(self) -> Any:
        """Create a fresh recognizer from the shared model."""

        try:
            from vosk import KaldiRecognizer
        except ImportError as exc:
            raise RuntimeError("Install vosk to use the Vosk speech engine.") from exc

        recognizer = KaldiRecognizer(self._load_model(), self.config.sample_rate)
        recognizer.SetWords(True)
        return recognizer

    def _load_model(self) -> Any:
        """Load the Vosk model lazily and reuse it."""

        with self._model_lock:
            if self._model is not None:
                return self._model
            if not self.config.model_path.exists():
                raise FileNotFoundError(
                    f"Vosk model folder was not found: {self.config.model_path}"
                )
            try:
                from vosk import Model, SetLogLevel
            except ImportError as exc:
                raise RuntimeError("Install vosk to use the Vosk speech engine.") from exc

            SetLogLevel(-1)
            self._model = Model(str(self.config.model_path))
            return self._model

    def _decode_audio_bytes(self, audio_bytes: bytes) -> tuple[bytes, float]:
        """Decode WAV bytes or pass through raw PCM bytes."""

        if not audio_bytes:
            return b"", 0.0
        if _looks_like_wav(audio_bytes):
            return _wav_to_pcm16_mono_16k(audio_bytes, self.config.sample_rate)
        duration = len(audio_bytes) / (self.config.sample_rate * PCM_SAMPLE_WIDTH)
        return audio_bytes, duration


class VoskRealtimeASRBuffer:
    """Streaming recognizer that emits text only when Vosk finalizes speech."""

    def __init__(self, service: VoskASRService) -> None:
        self.service = service
        self._recognizer: Any | None = None
        self._lock = threading.RLock()

    def clear(self) -> None:
        """Reset the live recognizer state."""

        with self._lock:
            self._recognizer = None

    def transcribe_chunk(self, audio_bytes: bytes) -> str:
        """Accept one PCM chunk and return finalized utterance text."""

        pcm_bytes, _ = self.service._decode_audio_bytes(audio_bytes)
        if not pcm_bytes:
            return ""

        with self._lock:
            recognizer = self._recognizer_or_create()
            if recognizer.AcceptWaveform(pcm_bytes):
                return _text_from_result(recognizer.Result())
            if self.service.config.return_partials:
                return _partial_from_result(recognizer.PartialResult())
            return ""

    def flush(self) -> str:
        """Return final text for any pending speech and reset the recognizer."""

        with self._lock:
            if self._recognizer is None:
                return ""
            text = _text_from_result(self._recognizer.FinalResult())
            self._recognizer = None
            return text

    def _recognizer_or_create(self) -> Any:
        if self._recognizer is None:
            self._recognizer = self.service.create_recognizer()
        return self._recognizer


def _looks_like_wav(audio_bytes: bytes) -> bool:
    return audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"


def _wav_to_pcm16_mono_16k(audio_bytes: bytes, target_rate: int) -> tuple[bytes, float]:
    with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        source_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        pcm_bytes = wav_file.readframes(frame_count)

    samples = _pcm_to_float32(pcm_bytes, sample_width=sample_width, channels=channels)
    if source_rate != target_rate:
        samples = _resample_linear(samples, source_rate, target_rate)
    samples = np.clip(samples, -1.0, 1.0)
    pcm16 = (samples * 32767.0).astype("<i2").tobytes()
    duration = len(samples) / target_rate if target_rate else 0.0
    return pcm16, duration


def _pcm_to_float32(pcm_bytes: bytes, *, sample_width: int, channels: int) -> np.ndarray:
    if channels <= 0:
        raise ValueError("Audio channel count must be greater than zero.")
    if sample_width not in {1, 2, 3, 4}:
        raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes.")
    if not pcm_bytes:
        return np.empty(0, dtype=np.float32)

    frame_width = sample_width * channels
    if len(pcm_bytes) % frame_width:
        pcm_bytes = pcm_bytes[: len(pcm_bytes) - (len(pcm_bytes) % frame_width)]

    if sample_width == 1:
        samples = (
            np.frombuffer(pcm_bytes, dtype=np.uint8).astype(np.float32) - 128.0
        ) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        samples = _pcm24_to_float32(pcm_bytes)
    else:
        samples = np.frombuffer(pcm_bytes, dtype="<i4").astype(np.float32) / 2147483648.0

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


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Audio sample rates must be greater than zero.")
    if samples.size == 0 or source_rate == target_rate:
        return samples.astype(np.float32, copy=False)

    target_size = max(1, int(round(samples.size / source_rate * target_rate)))
    source_positions = np.linspace(0.0, samples.size - 1, num=samples.size)
    target_positions = np.linspace(0.0, samples.size - 1, num=target_size)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _text_from_result(result_json: str) -> str:
    try:
        payload = json.loads(result_json)
    except json.JSONDecodeError:
        return ""
    return str(payload.get("text", "")).strip()


def _partial_from_result(result_json: str) -> str:
    try:
        payload = json.loads(result_json)
    except json.JSONDecodeError:
        return ""
    return str(payload.get("partial", "")).strip()


def _compact_spaces(text: str) -> str:
    return " ".join(text.split())


__all__ = [
    "VOSK_SAMPLE_RATE",
    "VoskASRConfig",
    "VoskASRService",
    "VoskRealtimeASRBuffer",
]
