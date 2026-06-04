"""Platform-specific speech-to-text engines.

Android uses the native SpeechRecognizer through pyjnius. Desktop keeps the
existing local ASR path behind lazy imports so Android never imports Vosk,
faster-whisper, or their native dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
import os
from pathlib import Path
from queue import Queue
import sys
import threading
from typing import Any


SpeechEvent = tuple[str, str]
SpeechEventSink = Callable[[SpeechEvent], None]


def is_android() -> bool:
    """Return True when running inside an Android Python runtime."""

    return sys.platform == "android" or "ANDROID_ARGUMENT" in os.environ


def language_code(language: str) -> str:
    """Map app language keys to recognizer locale strings."""

    return "fil-PH" if language == "tl" else "en-US"


class SpeechEngine(ABC):
    """Abstract speech engine used by the Flet speech tab."""

    def __init__(self, page: Any, state: Any, emit: SpeechEventSink) -> None:
        """Create a speech engine."""

        self.page = page
        self.state = state
        self.emit = emit

    @abstractmethod
    def start(self, language: str) -> None:
        """Start listening or recording."""

    @abstractmethod
    def stop(self) -> None:
        """Stop listening and finalize transcription."""

    def transcribe_bytes(self, audio_bytes: bytes) -> dict[str, str | float]:
        """Transcribe a complete audio payload when supported."""

        raise RuntimeError("File transcription is not supported on this platform.")


class AndroidSpeechEngine(SpeechEngine):
    """Android SpeechRecognizer implementation using pyjnius only."""

    def __init__(self, page: Any, state: Any, emit: SpeechEventSink) -> None:
        """Create an Android speech engine."""

        super().__init__(page, state, emit)
        self._recognizer: Any | None = None
        self._listener: Any | None = None
        self._activity: Any | None = None
        self._lock = threading.RLock()

    def start(self, language: str) -> None:
        """Start Android native speech recognition."""

        try:
            self._run_on_ui_thread(lambda: self._start_on_ui_thread(language_code(language)))
        except Exception as exc:
            self.emit(("error", f"Android speech recognizer failed: {exc}"))
            self.emit(("status", "stopped"))

    def stop(self) -> None:
        """Stop Android speech recognition."""

        def stop_recognizer() -> None:
            with self._lock:
                recognizer = self._recognizer
            if recognizer is not None:
                try:
                    recognizer.stopListening()
                except Exception:
                    try:
                        recognizer.cancel()
                    except Exception:
                        pass

        try:
            self._run_on_ui_thread(stop_recognizer)
        except Exception:
            self.emit(("status", "stopped"))

    def _start_on_ui_thread(self, locale: str) -> None:
        """Create and start SpeechRecognizer on Android's UI thread."""

        autoclass, PythonJavaClass, java_method = _load_jnius()
        activity = self._activity or _android_activity(autoclass)
        SpeechRecognizer = autoclass("android.speech.SpeechRecognizer")
        RecognizerIntent = autoclass("android.speech.RecognizerIntent")
        Intent = autoclass("android.content.Intent")

        if not SpeechRecognizer.isRecognitionAvailable(activity):
            raise RuntimeError("Speech recognition is not available on this device.")

        with self._lock:
            if self._recognizer is not None:
                try:
                    self._recognizer.destroy()
                except Exception:
                    pass
            self._activity = activity
            self._listener = _recognition_listener_class(
                PythonJavaClass,
                java_method,
                self.emit,
            )(self.emit)
            self._recognizer = SpeechRecognizer.createSpeechRecognizer(activity)
            self._recognizer.setRecognitionListener(self._listener)

            intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
            intent.putExtra(
                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM,
            )
            intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, locale)
            intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, locale)
            intent.putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, True)
            try:
                intent.putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, True)
            except Exception:
                pass
            self._recognizer.startListening(intent)
        self.emit(("status", "receiving"))

    def _run_on_ui_thread(self, callback: Callable[[], None]) -> None:
        """Run callback on Android UI thread when the helper is available."""

        try:
            from android.runnable import run_on_ui_thread

            run_on_ui_thread(callback)()
        except Exception:
            callback()


class WindowsSpeechEngine(SpeechEngine):
    """Desktop speech engine with lazy imports for existing local ASR logic."""

    def __init__(self, page: Any, state: Any, emit: SpeechEventSink) -> None:
        """Create a Windows/desktop speech engine."""

        super().__init__(page, state, emit)
        self._audio_chunks: Queue[bytes | None] = Queue()
        self._recorded_chunks: list[bytes] = []
        self._capture: Any | None = None
        self._worker_thread: threading.Thread | None = None
        self._active = False
        self._chunks_received = 0

    def start(self, language: str) -> None:
        """Start desktop microphone capture."""

        self._audio_chunks = Queue()
        self._recorded_chunks = []
        self._chunks_received = 0
        self._active = True
        self._worker_thread = threading.Thread(
            target=self._run_audio_worker,
            name="desktop-speech-engine",
            daemon=True,
        )
        self._worker_thread.start()
        try:
            from services.mic_capture import make_capture

            self._capture = make_capture(self.page)
            self._capture.start(self._submit_audio_chunk)
            self.emit(("status", "receiving"))
        except Exception as exc:
            self._capture = None
            self.emit(("error", f"Microphone capture failed: {exc}"))
            self._audio_chunks.put(None)

    def stop(self) -> None:
        """Stop desktop capture and finalize one transcription."""

        if not self._active:
            return
        self._active = False
        if self._capture is not None:
            try:
                self._capture.stop()
            except Exception:
                pass
            self._capture = None
        self._audio_chunks.put(None)

    def transcribe_bytes(self, audio_bytes: bytes) -> dict[str, str | float]:
        """Transcribe a complete audio payload on desktop."""

        return self._transcribe_with_existing_backend(audio_bytes)

    def _submit_audio_chunk(self, audio_bytes: bytes) -> None:
        """Receive microphone chunks from the desktop capture layer."""

        if self._active:
            self._audio_chunks.put(audio_bytes)

    def _run_audio_worker(self) -> None:
        """Collect chunks until Stop, then transcribe once."""

        try:
            while True:
                audio_bytes = self._audio_chunks.get()
                if audio_bytes is None:
                    break
                self._recorded_chunks.append(audio_bytes)
                self._chunks_received += 1
                if self._chunks_received == 1:
                    self.emit(("status", "receiving"))

            if self._recorded_chunks:
                self.emit(("status", "transcribing"))
                combined = b"".join(self._recorded_chunks)
                result = self.transcribe_bytes(combined)
                text = str(result.get("text", "")).strip()
                duration = float(result.get("duration", 0.0))
                if text:
                    self.emit(("result", text))
                    self.emit(("status", f"done:{duration:.1f}"))
                else:
                    self.emit(("status", "no_speech"))
            elif self._chunks_received == 0:
                self.emit(("status", "no_input"))
        except Exception as exc:
            self.emit(("error", str(exc)))
        finally:
            self._recorded_chunks.clear()
            self._active = False
            self.emit(("status", "stopped"))

    def _transcribe_with_existing_backend(self, audio_bytes: bytes) -> dict[str, str | float]:
        """Use existing desktop ASR when installed, then speech_recognition fallback."""

        try:
            return self._transcribe_with_local_backend(audio_bytes)
        except Exception as local_exc:
            try:
                return self._transcribe_with_speech_recognition(audio_bytes)
            except Exception as speech_exc:
                raise RuntimeError(
                    "Desktop speech recognition is unavailable. Install the "
                    "selected local ASR packages or speech_recognition. "
                    f"Local ASR error: {local_exc}; speech_recognition error: {speech_exc}"
                ) from speech_exc

    def _transcribe_with_local_backend(self, audio_bytes: bytes) -> dict[str, str | float]:
        """Lazy-load the existing Vosk or faster-whisper backend."""

        if self.state.settings.speech_engine == "vosk":
            from services.vosk_asr_service import VoskASRConfig, VoskASRService

            config = VoskASRConfig(
                model_path=self.state.settings.vosk_model_path(self.state.assets_dir),
                language=self.state.settings.speech_language,
                sample_rate=16_000,
                return_partials=False,
            )
            return VoskASRService(config).transcribe_bytes(audio_bytes)

        from services.asr_service import ASRConfig, ASRService

        config = ASRConfig(
            model_size_or_path=self.state.settings.whisper_model_name_or_path(
                self.state.assets_dir
            ),
            language=self.state.settings.speech_language,
            initial_prompt=self._language_prompt(),
            compute_type="int8",
            beam_size=1,
            vad_filter=True,
            mic_gain=self.state.settings.mic_gain,
            denoise_enabled=self.state.settings.denoise_enabled,
            highpass_enabled=self.state.settings.highpass_enabled,
            highpass_cutoff_hz=self.state.settings.highpass_cutoff_hz,
            denoise_prop_decrease=self.state.settings.denoise_prop_decrease,
            denoise_stationary=self.state.settings.denoise_stationary,
            no_speech_threshold=self.state.settings.no_speech_threshold,
            vad_silence_ms=self.state.settings.vad_silence_ms,
            vad_threshold=self.state.settings.vad_threshold,
        )
        return ASRService(config).transcribe_bytes(audio_bytes)

    def _transcribe_with_speech_recognition(self, audio_bytes: bytes) -> dict[str, str | float]:
        """Fallback desktop transcription using speech_recognition when present."""

        import speech_recognition as sr

        recognizer = sr.Recognizer()
        audio = sr.AudioData(audio_bytes, 16_000, 2)
        locale = language_code(self.state.settings.speech_language)
        try:
            text = recognizer.recognize_sphinx(audio, language=locale)
        except Exception:
            text = recognizer.recognize_google(audio, language=locale)
        duration = len(audio_bytes) / (16_000 * 2)
        return {
            "text": text,
            "language": self.state.settings.speech_language,
            "duration": round(duration, 3),
        }

    def _language_prompt(self) -> str | None:
        """Return a small language hint for Whisper."""

        if self.state.settings.speech_language == "tl":
            return (
                "Transcribe Filipino or Tagalog speech accurately. Keep Filipino "
                "words such as ako, ikaw, siya, tayo, salamat, mahal, oo, hindi, "
                "kumusta, nasa, para, akin, sayo, dito, doon, ngayon, bukas."
            )
        return "Transcribe clear English speech accurately."


def make_speech_engine(page: Any, state: Any, emit: SpeechEventSink) -> SpeechEngine:
    """Create the correct speech engine for the current platform."""

    if is_android():
        return AndroidSpeechEngine(page, state, emit)
    return WindowsSpeechEngine(page, state, emit)


def _load_jnius() -> tuple[Any, Any, Any]:
    """Import pyjnius helpers only inside Android speech code."""

    try:
        from jnius import PythonJavaClass, autoclass, java_method
    except Exception as exc:
        raise RuntimeError("pyjnius is required for Android speech recognition.") from exc
    return autoclass, PythonJavaClass, java_method


def _android_activity(autoclass: Any) -> Any:
    """Return the active Android activity for SpeechRecognizer."""

    for class_name in (
        "org.kivy.android.PythonActivity",
        "org.flet.app.MainActivity",
        "com.flet.app.MainActivity",
    ):
        try:
            activity = autoclass(class_name).mActivity
            if activity is not None:
                return activity
        except Exception:
            continue
    raise RuntimeError("Could not resolve the Android activity.")


def _recognition_listener_class(
    PythonJavaClass: Any,
    java_method: Any,
    emit: SpeechEventSink,
) -> type:
    """Build a pyjnius RecognitionListener class."""

    class _RecognitionListener(PythonJavaClass):
        __javainterfaces__ = ["android/speech/RecognitionListener"]
        __javacontext__ = "app"

        def __init__(self, callback: SpeechEventSink) -> None:
            super().__init__()
            self.callback = callback

        @java_method("(Landroid/os/Bundle;)V")
        def onReadyForSpeech(self, params: Any) -> None:
            self.callback(("status", "receiving"))

        @java_method("()V")
        def onBeginningOfSpeech(self) -> None:
            self.callback(("status", "receiving"))

        @java_method("(F)V")
        def onRmsChanged(self, rmsdB: float) -> None:
            return None

        @java_method("([B)V")
        def onBufferReceived(self, buffer: Any) -> None:
            return None

        @java_method("()V")
        def onEndOfSpeech(self) -> None:
            self.callback(("status", "transcribing"))

        @java_method("(I)V")
        def onError(self, error: int) -> None:
            self.callback(("error", _android_error_message(error)))
            self.callback(("status", "stopped"))

        @java_method("(Landroid/os/Bundle;)V")
        def onResults(self, results: Any) -> None:
            text = _android_result_text(results)
            if text:
                self.callback(("result", text))
                self.callback(("status", "done:0.0"))
            else:
                self.callback(("status", "no_speech"))
            self.callback(("status", "stopped"))

        @java_method("(Landroid/os/Bundle;)V")
        def onPartialResults(self, partialResults: Any) -> None:
            return None

        @java_method("(ILandroid/os/Bundle;)V")
        def onEvent(self, eventType: int, params: Any) -> None:
            return None

    return _RecognitionListener


def _android_result_text(bundle: Any) -> str:
    """Extract top recognition text from Android result bundle."""

    try:
        from jnius import autoclass

        SpeechRecognizer = autoclass("android.speech.SpeechRecognizer")
        matches = bundle.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
        if matches is not None and matches.size() > 0:
            return str(matches.get(0)).strip()
    except Exception:
        return ""
    return ""


def _android_error_message(error: int) -> str:
    """Return a user-facing Android SpeechRecognizer error."""

    messages = {
        1: "Network timeout.",
        2: "Network error.",
        3: "Audio recording error.",
        4: "Server error.",
        5: "Speech client error.",
        6: "No speech detected.",
        7: "No recognition match.",
        8: "Recognition service busy.",
        9: "Insufficient speech permissions.",
    }
    return messages.get(int(error), f"Android speech error {error}.")


__all__ = [
    "AndroidSpeechEngine",
    "SpeechEngine",
    "WindowsSpeechEngine",
    "is_android",
    "language_code",
    "make_speech_engine",
]
