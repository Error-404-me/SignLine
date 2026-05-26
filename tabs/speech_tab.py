"""Speech-to-text tab for offline Vosk or faster-whisper recognition."""

from __future__ import annotations

import asyncio
from pathlib import Path
from queue import Empty, Queue
import threading
import time

import flet as ft

from app_state import AppState
from components import build_output_card, copy_text, show_snack
from services.asr_service import ASRConfig, ASRService
from services.mic_capture import MicCapture, make_capture
from services.vosk_asr_service import (
    VoskASRConfig,
    VoskASRService,
)
from theme import (
    BORDER,
    CARD_RADIUS,
    CENTER_ALIGNMENT,
    PAGE_PADDING,
    PRIMARY_BLUE,
    SECTION_GAP,
    SURFACE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    border_all,
)


ASRUiEvent = tuple[str, str]


class SpeechTab:
    """Flet view and controller for live offline speech recognition."""

    def __init__(self, page: ft.Page, state: AppState) -> None:
        """Create speech tab controls."""

        self.page = page
        self.state = state
        self._events: Queue[ASRUiEvent] = Queue()
        self._audio_chunks: Queue[bytes | None] = Queue()
        self._service: ASRService | VoskASRService | None = None
        self._recorded_chunks: list[bytes] = []
        self._capture: MicCapture | None = None
        self._worker_thread: threading.Thread | None = None
        self._worker_active = False
        self._listening = False
        self._consumer_running = False
        self._animation_running = False
        self._started_at = 0.0
        self._audio_chunks_received = 0
        self._output_lines: list[str] = []
        self._last_result_text = ""
        self._last_result_at = 0.0
        self.file_picker = ft.FilePicker()
        self._register_file_picker()

        self.wave_bars = [
            ft.Container(
                width=10,
                height=22,
                bgcolor=PRIMARY_BLUE,
                border_radius=5,
                animate_size=ft.Animation(180, ft.AnimationCurve.EASE_IN_OUT),
            )
            for _ in range(5)
        ]
        self.mic_circle = ft.Container(
            width=136,
            height=136,
            border_radius=68,
            bgcolor="#DBEAFE",
            border=border_all(1, "#BFDBFE"),
            alignment=CENTER_ALIGNMENT,
            scale=1.0,
            animate_scale=ft.Animation(550, ft.AnimationCurve.EASE_IN_OUT),
            content=ft.Icon(ft.Icons.MIC, color=PRIMARY_BLUE, size=72),
        )
        self.status_text = ft.Text("Ready", size=15, weight=ft.FontWeight.W_600, color=TEXT_PRIMARY)
        self.timer_text = ft.Text("00:00", size=15, color=TEXT_MUTED)
        self.toggle_button = ft.Button(
            content=ft.Text("Start Listening", weight=ft.FontWeight.W_700),
            icon=ft.Icons.PLAY_ARROW,
            height=50,
            bgcolor=PRIMARY_BLUE,
            color="#FFFFFF",
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
            on_click=self._toggle_listening,
        )
        self.file_button = ft.OutlinedButton(
            content=ft.Text("Transcribe WAV"),
            icon=ft.Icons.AUDIO_FILE,
            height=50,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
            on_click=lambda _: self.page.run_task(self._pick_and_transcribe_audio),
        )
        self.output_text = ft.Text(
            "Recognised phrases will appear here.",
            selectable=True,
            size=14,
            color=TEXT_MUTED,
        )
        self._view = self._build_view()

    def _register_file_picker(self) -> None:
        """Attach the file picker service to the current Flet page."""

        try:
            self.page.register_service(self.file_picker)
        except Exception:
            pass

    def build(self) -> ft.Control:
        """Return the tab root control."""

        return self._view

    def on_visible(self) -> None:
        """Refresh visible state when the tab becomes active."""

        self.page.update()

    def on_hidden(self) -> None:
        """Stop audio chunk processing when leaving the tab."""

        self.stop()

    def submit_audio_chunk(self, audio_bytes: bytes) -> None:
        """Receive WAV or PCM bytes from the UI audio-capture layer."""

        if not self._listening:
            return
        self._audio_chunks.put(audio_bytes)

    def stop(self) -> None:
        """Request the speech worker to stop after flushing buffered audio."""

        if self._listening:
            self._listening = False
            self.status_text.value = "Stopping..."
            if self._capture:
                self._capture.stop()
                self._capture = None
            self._audio_chunks.put(None)
            self.page.update()

    def _build_view(self) -> ft.Control:
        """Create the responsive speech tab layout."""

        content = ft.Column(
            scroll=ft.ScrollMode.HIDDEN,
            spacing=SECTION_GAP,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            controls=[
                # ft.Text(
                #     "Convert your speech into text in real-time.",
                #     size=14,
                #     color=TEXT_MUTED,
                # ),
                ft.Container(
                    height=70,
                    bgcolor=SURFACE,
                    border=border_all(1, BORDER),
                    border_radius=CARD_RADIUS,
                    alignment=CENTER_ALIGNMENT,
                    content=ft.Row(
                        spacing=8,
                        alignment=ft.MainAxisAlignment.CENTER,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=self.wave_bars,
                    ),
                ),
                ft.Container(alignment=CENTER_ALIGNMENT, content=self.mic_circle),
                ft.Row(
                    alignment=ft.MainAxisAlignment.CENTER,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[self.status_text, ft.Text("-", color=TEXT_MUTED), self.timer_text],
                ),
                ft.Row(
                    controls=[self.toggle_button, self.file_button],
                    wrap=True,
                    spacing=10,
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                build_output_card(
                    title="Text Output",
                    text_control=self.output_text,
                    on_copy=lambda: copy_text(self.page, "\n".join(self._output_lines)),
                    on_clear=self._clear_output,
                    extra_actions=[
                        ft.IconButton(
                            icon=ft.Icons.VOLUME_UP,
                            icon_color=PRIMARY_BLUE,
                            tooltip="Speak output",
                            on_click=lambda _: self.state.tts.speak(
                                "\n".join(self._output_lines)
                            ),
                        )
                    ],
                    accent=PRIMARY_BLUE,
                    height=190,
                ),
            ],
        )

        return ft.SafeArea(
            content=ft.Container(
                expand=True,
                padding=PAGE_PADDING,
                content=ft.ResponsiveRow(
                    alignment=ft.MainAxisAlignment.CENTER,
                    controls=[
                        ft.Container(
                            col={"xs": 12, "sm": 10, "md": 8, "lg": 6},
                            content=content,
                        )
                    ],
                ),
            )
        )

    def _toggle_listening(self, _: ft.ControlEvent) -> None:
        """Handle the start/stop button."""

        if self._listening:
            self.stop()
            return
        self._start_listening()

    def _start_listening(self) -> None:
        """Start recording audio chunks for transcription on Stop."""

        self._service = None
        self._recorded_chunks = []
        worker_name = f"{self.state.settings.speech_engine}-asr-recording"
        self._events = Queue()
        self._audio_chunks = Queue()
        self._listening = True
        self._worker_active = True
        self._audio_chunks_received = 0
        self._started_at = time.monotonic()
        self.status_text.value = "Recording..."
        self.timer_text.value = "00:00"
        if not self._output_lines:
            self.output_text.value = "Recording audio. Press Stop to transcribe."
            self.output_text.color = TEXT_MUTED
        self.toggle_button.content = ft.Text("Stop Listening", weight=ft.FontWeight.W_700)
        self.toggle_button.icon = ft.Icons.STOP
        self.toggle_button.bgcolor = "#1D4ED8"
        self.toggle_button.disabled = False
        self._worker_thread = threading.Thread(
            target=self._run_audio_worker,
            name=worker_name,
            daemon=True,
        )
        self._worker_thread.start()
        try:
            self._capture = make_capture(self.page)
            self._capture.start(self.submit_audio_chunk)
        except Exception as exc:
            self._capture = None
            self._events.put(("error", f"Microphone capture failed: {exc}"))
            self._audio_chunks.put(None)
        if not self._consumer_running:
            self.page.run_task(self._consume_events)
        self.page.run_task(self._watch_for_audio_input)
        if not self._animation_running:
            self.page.run_task(self._animate_listening)
        self.page.update()

    async def _watch_for_audio_input(self) -> None:
        """Show a clear status if no UI capture layer sends audio chunks."""

        await asyncio.sleep(2.5)
        if self._listening and self._audio_chunks_received == 0:
            self._events.put(("status", "no_input"))

    async def _pick_and_transcribe_audio(self) -> None:
        """Pick a WAV file and transcribe it with the same byte-only ASR API."""

        files = await self.file_picker.pick_files(
            dialog_title="Choose a WAV file",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["wav"],
            allow_multiple=False,
            with_data=True,
        )
        if not files:
            return

        selected = files[0]
        audio_bytes = selected.bytes
        if audio_bytes is None and selected.path:
            audio_bytes = await asyncio.to_thread(Path(selected.path).read_bytes)
        if audio_bytes is None:
            show_snack(self.page, "Could not read selected audio.", bgcolor="#B91C1C")
            return

        self.file_button.disabled = True
        self.status_text.value = "Transcribing..."
        self.page.update()
        try:
            service = self._service or self._build_current_asr_service()
            result = await asyncio.to_thread(service.transcribe_bytes, audio_bytes)
            text = str(result.get("text", "")).strip()
            if text:
                self._append_result(text)
                self.status_text.value = f"{result.get('duration', 0.0)}s audio"
            else:
                self.output_text.value = "No speech detected in selected audio."
                self.output_text.color = TEXT_MUTED
                self.status_text.value = "No speech detected"
        except Exception as exc:
            self.status_text.value = "Error"
            show_snack(self.page, f"Audio transcription failed: {exc}", bgcolor="#B91C1C")
        finally:
            self.file_button.disabled = False
            self.page.update()

    def _build_asr_service(self) -> ASRService:
        """Create a faster-whisper service from current settings."""

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
        return ASRService(config)

    def _build_vosk_service(self) -> VoskASRService:
        """Create a Vosk service from current settings."""

        config = VoskASRConfig(
            model_path=self.state.settings.vosk_model_path(self.state.assets_dir),
            language=self.state.settings.speech_language,
            sample_rate=16_000,
            return_partials=False,
        )
        return VoskASRService(config)

    def _build_current_asr_service(self) -> ASRService | VoskASRService:
        """Create the currently selected speech service."""

        if self.state.settings.speech_engine == "vosk":
            return self._build_vosk_service()
        return self._build_asr_service()

    def _language_prompt(self) -> str | None:
        """Return a small language hint for Whisper."""

        if self.state.settings.speech_language == "tl":
            return (
                "Transcribe Filipino or Tagalog speech accurately. Keep Filipino "
                "words such as ako, ikaw, siya, tayo, salamat, mahal, oo, hindi, "
                "kumusta, nasa, para, akin, sayo, dito, doon, ngayon, bukas."
            )
        return "Transcribe clear English speech accurately."

    def _run_audio_worker(self) -> None:
        """Record chunks, then transcribe the full buffer after Stop."""

        try:
            while True:
                audio_bytes = self._audio_chunks.get()
                if audio_bytes is None:
                    break
                self._recorded_chunks.append(audio_bytes)
                self._audio_chunks_received += 1
                if self._audio_chunks_received == 1:
                    self._events.put(("status", "receiving"))

            if self._recorded_chunks:
                self._events.put(("status", "transcribing"))
                combined = b"".join(self._recorded_chunks)
                try:
                    service = self._build_current_asr_service()
                    result = service.transcribe_bytes(combined)
                    text = str(result.get("text", "")).strip()
                    duration = float(result.get("duration", 0.0))
                    if text:
                        self._events.put(("result", text))
                        self._events.put(("status", f"done:{duration:.1f}"))
                    else:
                        self._events.put(("status", "no_speech"))
                except Exception as exc:
                    self._events.put(("error", str(exc)))
        except Exception as exc:
            self._events.put(("error", f"Speech recognition stopped: {exc}"))
        finally:
            self._recorded_chunks.clear()
            self._events.put(("status", "stopped"))
            self._worker_active = False

    async def _consume_events(self) -> None:
        """Consume ASR worker events on the Flet event loop."""

        self._consumer_running = True
        try:
            while self._listening or self._worker_active or not self._events.empty():
                try:
                    event = await asyncio.to_thread(self._events.get, True, 0.2)
                except Empty:
                    continue
                self._handle_event(event)
                self.page.update()
        finally:
            self._consumer_running = False

    async def _animate_listening(self) -> None:
        """Animate waveform bars and the microphone pulse while listening."""

        self._animation_running = True
        frames = [
            [24, 38, 56, 34, 26],
            [44, 28, 50, 62, 32],
            [30, 60, 36, 48, 54],
            [52, 36, 28, 58, 40],
        ]
        index = 0
        try:
            while self._listening:
                heights = frames[index % len(frames)]
                for bar, height in zip(self.wave_bars, heights, strict=True):
                    bar.height = height
                    bar.opacity = 1.0
                elapsed = int(time.monotonic() - self._started_at)
                self.timer_text.value = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
                self.mic_circle.scale = 1.08 if index % 2 else 1.0
                self.page.update()
                index += 1
                await asyncio.sleep(0.28)
        finally:
            for bar in self.wave_bars:
                bar.height = 22
                bar.opacity = 0.65
            self.mic_circle.scale = 1.0
            self._animation_running = False
            self.page.update()

    def _handle_event(self, event: ASRUiEvent) -> None:
        """Apply an ASR service event to the UI."""

        kind, message = event
        if kind == "result":
            self._append_result(message)
        elif kind == "error":
            show_snack(self.page, message, bgcolor="#B91C1C")
            self.status_text.value = "Error"
        elif kind == "status":
            if message == "receiving":
                self.status_text.value = "Recording..."
                if not self._output_lines:
                    self.output_text.value = "Recording audio. Press Stop to transcribe."
                    self.output_text.color = TEXT_MUTED
            elif message == "transcribing":
                self.status_text.value = "Transcribing..."
                self.toggle_button.disabled = True
            elif message == "no_input":
                self.status_text.value = "No audio input connected"
                if not self._output_lines:
                    self.output_text.value = (
                        "No audio chunks received. Use Transcribe WAV or connect "
                        "a UI audio capture layer."
                    )
                    self.output_text.color = TEXT_MUTED
            elif message == "no_speech":
                self.status_text.value = "No speech detected"
                self.toggle_button.disabled = False
            elif message.startswith("done:"):
                duration_text = message.split(":", 1)[1]
                self.status_text.value = f"Transcribed {duration_text}s"
                self.toggle_button.disabled = False
            elif message == "stopped":
                self._listening = False
                self.status_text.value = "Ready"
                self.toggle_button.content = ft.Text("Start Listening", weight=ft.FontWeight.W_700)
                self.toggle_button.icon = ft.Icons.PLAY_ARROW
                self.toggle_button.bgcolor = PRIMARY_BLUE
                self.toggle_button.disabled = False

    def _append_result(self, text: str) -> None:
        """Append recognised speech text to visible output and history."""

        cleaned = text.strip()
        if not cleaned:
            return
        if self._looks_like_whisper_artifact(cleaned):
            return
        now = time.monotonic()
        if cleaned == self._last_result_text and now - self._last_result_at < 6.0:
            return
        self._output_lines.append(cleaned)
        self.output_text.value = "\n".join(self._output_lines)
        self.output_text.color = TEXT_PRIMARY
        self.state.record_text(cleaned, "speech")
        self._last_result_text = cleaned
        self._last_result_at = now

    def _looks_like_whisper_artifact(self, text: str) -> bool:
        """Filter common tiny/base Whisper hallucinations from silence."""

        if self.state.settings.speech_engine != "whisper":
            return False
        lowered = text.lower().strip(" .!?")
        if lowered in {"", "how are you", "thank you", "thanks"}:
            return True
        artifact_prefixes = (
            "thank you for watching",
            "please subscribe",
            "thanks for watching",
        )
        return any(lowered.startswith(prefix) for prefix in artifact_prefixes)

    def _clear_output(self) -> None:
        """Clear visible speech output."""

        self._output_lines.clear()
        self.output_text.value = "Recognised phrases will appear here."
        self.output_text.color = TEXT_MUTED
        self.page.update()


__all__ = ["SpeechTab"]
