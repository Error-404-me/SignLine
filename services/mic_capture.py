"""Microphone capture adapters for speech-to-text input."""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import inspect
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
        self.page = page
        self._recorder: Any | None = None

    def start(self, on_chunk: AudioChunkCallback) -> None:
        """Add a Flet recorder to the page and start PCM streaming."""

        import flet as ft

        recorder_class = getattr(ft, "AudioRecorder", None)
        if recorder_class is None:
            raise RuntimeError("This Flet build does not provide ft.AudioRecorder.")

        audio_encoder = getattr(getattr(ft, "AudioEncoder", object), "PCM_16BIT", "PCM_16BIT")

        def on_data(event: Any) -> None:
            encoded = getattr(event, "data", "") or ""
            if not encoded:
                return
            on_chunk(base64.b64decode(encoded))

        self._recorder = recorder_class(
            audio_encoder=audio_encoder,
            sample_rate=16_000,
            num_channels=1,
            on_data=on_data,
        )
        self.page.overlay.append(self._recorder)
        self.page.update()
        _call_maybe_async(self._recorder.start_recording, self.page)

    def stop(self) -> None:
        """Stop recording and remove the recorder from page overlay."""

        if self._recorder is None:
            return
        try:
            _call_maybe_async(self._recorder.stop_recording, self.page)
        finally:
            if self._recorder in self.page.overlay:
                self.page.overlay.remove(self._recorder)
                self.page.update()
            self._recorder = None


def make_capture(page: Any | None = None) -> MicCapture:
    """Return the microphone capture implementation for the current platform."""

    if platform.system() in {"Windows", "Darwin", "Linux"}:
        return DesktopMicCapture()
    return FletMicCapture(page)


def _call_maybe_async(function: Callable[..., Any], page: Any) -> None:
    result = function()
    if inspect.isawaitable(result):
        async def runner() -> None:
            await result

        page.run_task(runner)


__all__ = [
    "DesktopMicCapture",
    "FletMicCapture",
    "MicCapture",
    "make_capture",
]
