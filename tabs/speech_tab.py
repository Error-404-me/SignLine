"""Speech-to-text tab backed by platform STT services."""

from __future__ import annotations

import asyncio
import time

import flet as ft
from flet_permission_handler import PermissionHandler

from app_state import AppState
from components import build_output_card, copy_text, show_snack
from services.stt_service import STTResult, STTService, make_stt_service
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


class SpeechTab:
    """Flet view and controller for platform speech recognition."""

    def __init__(self, page: ft.Page, state: AppState) -> None:
        """Create speech tab controls."""

        self.page = page
        self.state = state
        self._stt: STTService | None = None
        self._permission_handler = PermissionHandler()
        self._register_permission_handler()
        self._listening = False
        self._animation_running = False
        self._started_at = 0.0
        self._output_lines: list[str] = []
        self._last_result_text = ""
        self._last_result_at = 0.0

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
        self.status_text = ft.Text(
            "Ready",
            size=15,
            weight=ft.FontWeight.W_600,
            color=TEXT_PRIMARY,
        )
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
        self.output_text = ft.Text(
            "Recognised phrases will appear here.",
            selectable=True,
            size=14,
            color=TEXT_MUTED,
        )
        self._view = self._build_view()

    def _register_permission_handler(self) -> None:
        """Register microphone permission handler with the page."""

        try:
            self.page.overlay.append(self._permission_handler)
        except Exception:
            try:
                self.page.register_service(self._permission_handler)
            except Exception:
                pass

    def build(self) -> ft.Control:
        """Return the tab root control."""

        return self._view

    def on_visible(self) -> None:
        """Refresh visible state when the tab becomes active."""

        self.page.update()

    def on_hidden(self) -> None:
        """Stop recognition when leaving the tab."""

        self.stop()

    def stop(self) -> None:
        """Stop the active STT service and reset controls."""

        if self._listening:
            self._listening = False
            if self._stt:
                self._stt.stop()
                self._stt = None
            self.status_text.value = "Ready"
            self.toggle_button.content = ft.Text(
                "Start Listening",
                weight=ft.FontWeight.W_700,
            )
            self.toggle_button.icon = ft.Icons.PLAY_ARROW
            self.toggle_button.bgcolor = PRIMARY_BLUE
            self.page.update()

    def _build_view(self) -> ft.Control:
        """Create the responsive speech tab layout."""

        content = ft.Column(
            scroll=ft.ScrollMode.HIDDEN,
            spacing=SECTION_GAP,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            controls=[
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
                    controls=[
                        self.status_text,
                        ft.Text("-", color=TEXT_MUTED),
                        self.timer_text,
                    ],
                ),
                ft.Row(
                    controls=[self.toggle_button],
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
        self.page.run_task(self._start_with_permission_check)

    async def _start_with_permission_check(self) -> None:
        """Request microphone permission before starting STT."""

        from services.permission_service import ensure_microphone_permission

        granted = await ensure_microphone_permission(self.page, self._permission_handler)
        if not granted:
            show_snack(
                self.page,
                "Microphone permission is required.",
                bgcolor="#B91C1C",
            )
            return
        self._start_listening()

    def _start_listening(self) -> None:
        """Start platform speech recognition."""

        from services.asr_service import ASRConfig

        config = ASRConfig(
            model_size_or_path=self.state.settings.whisper_model_name_or_path(
                self.state.assets_dir
            ),
            language=self.state.settings.speech_language,
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
        self._stt = make_stt_service(asr_config=config)
        if not self._stt.is_available():
            show_snack(
                self.page,
                "Speech recognition is not available on this device.",
                bgcolor="#B91C1C",
            )
            return

        self._listening = True
        self._started_at = time.monotonic()
        self.status_text.value = "Listening..."
        self.timer_text.value = "00:00"
        self.toggle_button.content = ft.Text(
            "Stop Listening",
            weight=ft.FontWeight.W_700,
        )
        self.toggle_button.icon = ft.Icons.STOP
        self.toggle_button.bgcolor = "#1D4ED8"
        if not self._animation_running:
            self.page.run_task(self._animate_listening)
        self._stt.start(self._on_stt_result)
        self.page.update()

    def _on_stt_result(self, result: STTResult) -> None:
        """Apply one STT result to output or live status."""

        if not result.text:
            return
        if result.is_final:
            self._append_result(result.text)
            self.status_text.value = "Listening..." if self._listening else "Ready"
        else:
            preview = result.text[:40] + "..." if len(result.text) > 40 else result.text
            self.status_text.value = f"Hearing: {preview}"
        self.page.update()

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
