"""Microphone capture adapters for speech-to-text input."""

from __future__ import annotations

from abc import ABC, abstractmethod
import platform
from typing import Callable, Any

import numpy as np


AudioChunkCallback = Callable[[bytes], None]


class MicCapture(ABC):
    """Abstract microphone capture interface."""

    @abstractmethod
    def start(self, on_chunk: AudioChunkCallback) -> None:
        """Start capture and call ``on_chunk`` with raw PCM bytes."""

    @abstractmethod
    def stop(self) -> None:
        """Stop capture and release any recorder resources."""


class DesktopMicCapture(MicCapture):
    """Desktop microphone capture using sounddevice."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        blocksize: int = 4_000,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self._stream: Any | None = None

    def start(self, on_chunk: AudioChunkCallback) -> None:
        """Open the default desktop microphone and stream PCM chunks."""

        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "Install sounddevice to use desktop microphone capture."
            ) from exc

        def callback(indata: np.ndarray, *_: Any) -> None:
            samples = np.clip(indata.reshape(-1), -1.0, 1.0)
            pcm_bytes = (samples * 32767.0).astype("<i2").tobytes()
            on_chunk(pcm_bytes)

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop and close the sounddevice stream."""

        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None


class FletMicCapture(MicCapture):
    """Mobile microphone capture using Flet AudioRecorder."""

    def __init__(self, page: Any) -> None:
        if page is None:
            raise ValueError("FletMicCapture requires a Flet page.")
        self._page = page
        self._recorder: Any | None = None
        self._on_chunk: AudioChunkCallback | None = None

    def start(self, on_chunk: AudioChunkCallback) -> None:
        """Add a Flet recorder to the page and start PCM streaming."""

        import base64
        import flet as ft

        if getattr(ft, "AudioRecorder", None) is None:
            raise RuntimeError("This Flet build does not provide ft.AudioRecorder.")

        self._on_chunk = on_chunk
        self._recorder = ft.AudioRecorder(
            audio_encoder=ft.AudioEncoder.PCM_16BIT,
            sample_rate=16_000,
            num_channels=1,
            on_data=self._handle_data,
        )
        self._page.overlay.append(self._recorder)
        self._page.update()
        self._recorder.start_recording()

    def _handle_data(self, event: Any) -> None:
        """Decode base64 PCM chunks from Flet AudioRecorder."""

        if self._on_chunk and getattr(event, "data", None):
            import base64

            self._on_chunk(base64.b64decode(event.data))

    def stop(self) -> None:
        """Stop recording and remove the recorder from page overlay."""

        if self._recorder is None:
            return
        try:
            self._recorder.stop_recording()
        finally:
            try:
                self._page.overlay.remove(self._recorder)
                self._page.update()
            except Exception:
                pass
            self._recorder = None
            self._on_chunk = None


def make_capture(page: Any | None = None) -> MicCapture:
    """Return the microphone capture implementation for the current platform."""

    from services.stt_service import _is_android

    if _is_android():
        return FletMicCapture(page)
    if platform.system() in {"Windows", "Darwin", "Linux"}:
        return DesktopMicCapture()
    return FletMicCapture(page)


__all__ = [
    "DesktopMicCapture",
    "FletMicCapture",
    "MicCapture",
    "make_capture",
]
