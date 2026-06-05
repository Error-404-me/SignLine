"""Platform-specific speech-to-text services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


@dataclass
class STTResult:
    """One speech-to-text result emitted by an STT backend."""

    text: str
    is_final: bool
    confidence: float = 1.0


class STTService(ABC):
    """Abstract speech-to-text interface."""

    @abstractmethod
    def start(self, on_result: Callable[[STTResult], None]) -> None:
        """Start listening and emit recognition results."""

    @abstractmethod
    def stop(self) -> None:
        """Stop listening and release recognizer resources."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether this STT backend can run on the current platform."""


class AndroidSTTService(STTService):
    """Uses Android SpeechRecognizer via pyjnius==1.6.1 for on-device STT."""

    def __init__(self, language: str = "en") -> None:
        self._on_result: Callable[[STTResult], None] | None = None
        self._recognizer = None
        self._listener = None
        self._listening = False
        self._language = _android_locale(language)

    def is_available(self) -> bool:
        """Return whether Android native speech recognition is available."""

        try:
            from jnius import autoclass

            SpeechRecognizer = autoclass("android.speech.SpeechRecognizer")
            context = autoclass("org.kivy.android.PythonActivity").mActivity
            return bool(SpeechRecognizer.isRecognitionAvailable(context))
        except Exception:
            return False

    def start(self, on_result: Callable[[STTResult], None]) -> None:
        """Start Android SpeechRecognizer on the Android UI thread."""

        try:
            from android.runnable import run_on_ui_thread
            from jnius import PythonJavaClass, autoclass, java_method

            self._on_result = on_result

            Intent = autoclass("android.content.Intent")
            RecognizerIntent = autoclass("android.speech.RecognizerIntent")
            SpeechRecognizer = autoclass("android.speech.SpeechRecognizer")
            context = autoclass("org.kivy.android.PythonActivity").mActivity

            service = self

            class RecognitionListener(PythonJavaClass):
                __javainterfaces__ = ["android/speech/RecognitionListener"]
                __javacontext__ = "app"

                def __init__(self, callback: Callable[[STTResult], None]) -> None:
                    self.callback = callback
                    super().__init__()

                @java_method("(Landroid/os/Bundle;)V")
                def onReadyForSpeech(self, params) -> None:
                    return None

                @java_method("(Landroid/os/Bundle;)V")
                def onResults(self, results) -> None:
                    text = _bundle_first_text(results, RecognizerIntent, SpeechRecognizer)
                    if text:
                        self.callback(STTResult(text=text, is_final=True))
                    service._listening = False

                @java_method("(Landroid/os/Bundle;)V")
                def onPartialResults(self, results) -> None:
                    text = _bundle_first_text(results, RecognizerIntent, SpeechRecognizer)
                    if text:
                        self.callback(STTResult(text=text, is_final=False))

                @java_method("(I)V")
                def onError(self, error) -> None:
                    service._listening = False

                @java_method("()V")
                def onEndOfSpeech(self) -> None:
                    return None

                @java_method("(F)V")
                def onRmsChanged(self, rmsdB) -> None:
                    return None

                @java_method("([B)V")
                def onBufferReceived(self, buffer) -> None:
                    return None

                @java_method("()V")
                def onBeginningOfSpeech(self) -> None:
                    return None

                @java_method("(ILandroid/os/Bundle;)V")
                def onEvent(self, eventType, params) -> None:
                    return None

            @run_on_ui_thread
            def _start_on_ui() -> None:
                if self._recognizer is not None:
                    try:
                        self._recognizer.destroy()
                    except Exception:
                        pass
                self._recognizer = SpeechRecognizer.createSpeechRecognizer(context)
                self._listener = RecognitionListener(on_result)
                self._recognizer.setRecognitionListener(self._listener)
                intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
                intent.putExtra(
                    RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                    RecognizerIntent.LANGUAGE_MODEL_FREE_FORM,
                )
                intent.putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, True)
                intent.putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
                intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, self._language)
                intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, self._language)
                try:
                    intent.putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, True)
                except Exception:
                    pass
                self._recognizer.startListening(intent)
                self._listening = True

            _start_on_ui()

        except Exception as exc:
            print(f"[AndroidSTT] start failed: {exc}")

    def stop(self) -> None:
        """Stop Android SpeechRecognizer on the Android UI thread."""

        try:
            from android.runnable import run_on_ui_thread

            @run_on_ui_thread
            def _stop_on_ui() -> None:
                if self._recognizer:
                    try:
                        self._recognizer.stopListening()
                    except Exception:
                        pass
                    try:
                        self._recognizer.destroy()
                    except Exception:
                        pass
                    self._recognizer = None
                    self._listener = None
                self._listening = False

            _stop_on_ui()
        except Exception as exc:
            print(f"[AndroidSTT] stop failed: {exc}")


class DesktopSTTService(STTService):
    """Uses faster-whisper on desktop via existing ASRService."""

    def __init__(self, asr_config) -> None:
        self._config = asr_config
        self._service = None
        self._on_result: Callable[[STTResult], None] | None = None
        self._recording = False
        self._chunks: list[bytes] = []
        self._capture = None

    def is_available(self) -> bool:
        """Return whether faster-whisper is importable."""

        try:
            from faster_whisper import WhisperModel  # noqa: F401

            return True
        except ImportError:
            return False

    def start(self, on_result: Callable[[STTResult], None]) -> None:
        """Start desktop capture and buffer chunks until stop."""

        from services.asr_service import ASRService
        from services.mic_capture import make_capture

        self._service = ASRService(self._config)
        self._on_result = on_result
        self._chunks = []
        self._recording = True
        self._capture = make_capture()
        self._capture.start(self._on_chunk)

    def _on_chunk(self, audio_bytes: bytes) -> None:
        """Collect desktop audio chunks while recording."""

        if self._recording:
            self._chunks.append(audio_bytes)

    def stop(self) -> None:
        """Stop capture and transcribe buffered audio in a daemon thread."""

        self._recording = False
        if self._capture:
            self._capture.stop()
            self._capture = None
        if self._chunks and self._service and self._on_result:
            import threading

            threading.Thread(target=self._transcribe, daemon=True).start()

    def _transcribe(self) -> None:
        """Run desktop transcription in the background."""

        try:
            combined = b"".join(self._chunks)
            self._chunks = []
            result = self._service.transcribe_bytes(combined)
            text = str(result.get("text", "")).strip()
            if text and self._on_result:
                self._on_result(STTResult(text=text, is_final=True))
        except Exception as exc:
            print(f"[DesktopSTT] transcribe failed: {exc}")


class NoOpSTTService(STTService):
    """Silent fallback when no STT engine is available."""

    def start(self, on_result) -> None:
        """Start no-op recognition."""

    def stop(self) -> None:
        """Stop no-op recognition."""

    def is_available(self) -> bool:
        """Return False because this is a fallback."""

        return False


def _is_android() -> bool:
    """Return True when running on Android."""

    try:
        from plyer.utils import platform as plyer_platform

        return str(plyer_platform).lower() == "android"
    except Exception:
        import sys

        return hasattr(sys, "getandroidapilevel")


def make_stt_service(asr_config=None) -> STTService:
    """Return the right STT backend for the current platform."""

    if _is_android():
        language = getattr(asr_config, "language", None) or "en"
        svc = AndroidSTTService(language=language)
        if svc.is_available():
            return svc

    if asr_config is not None:
        svc = DesktopSTTService(asr_config)
        if svc.is_available():
            return svc

    return NoOpSTTService()


def _android_locale(language: str | None) -> str:
    """Map app language keys to Android recognizer locales."""

    return "fil-PH" if language in {"tl", "fil", "fil-PH"} else "en-US"


def _bundle_first_text(results, RecognizerIntent, SpeechRecognizer) -> str:
    """Extract the first recognized phrase from an Android result bundle."""

    keys = [
        getattr(SpeechRecognizer, "RESULTS_RECOGNITION", None),
        getattr(RecognizerIntent, "EXTRA_RESULTS", None),
        "android.speech.extra.PARTIAL_RESULTS",
    ]
    for key in keys:
        if not key:
            continue
        try:
            matches = results.getStringArrayList(key)
            if matches and matches.size() > 0:
                text = matches.get(0)
                if text:
                    return str(text)
        except Exception:
            continue
    return ""


__all__ = [
    "STTResult",
    "STTService",
    "AndroidSTTService",
    "DesktopSTTService",
    "NoOpSTTService",
    "make_stt_service",
]
