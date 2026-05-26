"""Cross-platform text-to-speech service for Flet mobile and desktop."""

from __future__ import annotations

import threading


class TTSService:
    """Speak text without blocking the UI, using plyer before pyttsx3."""

    def __init__(self) -> None:
        """Create the TTS service."""

        self._is_speaking = False
        self._lock = threading.RLock()

    @property
    def is_speaking(self) -> bool:
        """Return whether a speech thread is currently active."""

        with self._lock:
            return self._is_speaking

    def speak(self, text: str) -> None:
        """Speak text on a daemon thread so the UI never blocks."""

        cleaned = text.strip()
        if not cleaned:
            return
        thread = threading.Thread(
            target=self._speak_thread,
            args=(cleaned,),
            name="tts-speak",
            daemon=True,
        )
        with self._lock:
            self._is_speaking = True
        thread.start()

    def _speak(self, text: str) -> None:
        """Speak with plyer first, then pyttsx3 as a desktop fallback."""

        try:
            from plyer import tts

            tts.speak(text)
            return
        except Exception:
            pass

        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

    def stop(self) -> None:
        """No-op because plyer does not support mid-speech stop."""

        pass

    def _speak_thread(self, text: str) -> None:
        """Run the blocking TTS call and reset speaking state."""

        try:
            self._speak(text)
        finally:
            with self._lock:
                self._is_speaking = False


__all__ = ["TTSService"]
