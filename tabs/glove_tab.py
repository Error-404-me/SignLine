"""Bluetooth glove tab for ESP32 flex sensors and BNO055 IMU input."""

from __future__ import annotations

import asyncio
from queue import Empty, Queue
import time
from typing import Any

import flet as ft

from app_state import AppState
from components import build_output_card, copy_text, show_snack
from services.glove_parser import GloveReading
from theme import (
    BORDER,
    CARD_RADIUS,
    PAGE_PADDING,
    PRIMARY_BLUE,
    PRIMARY_GREEN,
    SECTION_GAP,
    SURFACE,
    SURFACE_MUTED,
    TEXT_MUTED,
    TEXT_PRIMARY,
    border_all,
)


class GloveTab:
    """Flet view/controller for Bluetooth glove letters and sensor readings."""

    def __init__(self, page: ft.Page, state: AppState) -> None:
        """Create glove tab controls."""

        self.page = page
        self.state = state
        self._events: Queue[tuple[str, Any]] = Queue()
        self._visible = False
        self._consumer_running = False
        self._watcher_running = False
        self._current_word = ""
        self._last_letter_at = 0.0
        self._completed_words: list[str] = []
        self._sensor_values: dict[str, ft.Text] = {}

        self.status_dot = ft.Container(width=10, height=10, border_radius=5, bgcolor="#DC2626")
        self.status_text = ft.Text("Not connected", size=14, color=TEXT_MUTED, weight=ft.FontWeight.W_600)
        self.connect_button = ft.Button(
            "Connect",
            icon=ft.Icons.BLUETOOTH,
            height=40,
            bgcolor=PRIMARY_BLUE,
            color="#FFFFFF",
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
            on_click=self._connect_clicked,
        )
        self.current_letter = ft.Text(
            "-",
            size=80,
            weight=ft.FontWeight.W_800,
            color=PRIMARY_GREEN,
            text_align=ft.TextAlign.CENTER,
        )
        self.current_word = ft.Text(
            "",
            size=24,
            weight=ft.FontWeight.W_500,
            color=TEXT_PRIMARY,
            text_align=ft.TextAlign.CENTER,
        )
        self.output_text = ft.Text(
            "Completed glove words will appear here.",
            selectable=True,
            size=14,
            color=TEXT_MUTED,
        )
        self.auto_speak_switch = ft.Switch(
            label="Auto-speak",
            value=self.state.settings.auto_speak,
            active_color=PRIMARY_GREEN,
            on_change=self._auto_speak_changed,
        )
        self._view = self._build_view()

    def build(self) -> ft.Control:
        """Return the tab root control."""

        return self._view

    def on_visible(self) -> None:
        """Register Bluetooth callbacks when the tab becomes active."""

        self._visible = True
        self.auto_speak_switch.value = self.state.settings.auto_speak
        self.state.bluetooth.on_data = self._handle_bluetooth_data
        self.state.bluetooth.on_reading = self._handle_bluetooth_reading
        self._refresh_connection_status()
        if not self._consumer_running:
            self.page.run_task(self._consume_events)
        if not self._watcher_running:
            self.page.run_task(self._watch_silence)
        self.page.update()

    def on_hidden(self) -> None:
        """Unregister Bluetooth callbacks when leaving the tab."""

        self._visible = False
        if self.state.bluetooth.on_data == self._handle_bluetooth_data:
            self.state.bluetooth.on_data = None
        if self.state.bluetooth.on_reading == self._handle_bluetooth_reading:
            self.state.bluetooth.on_reading = None

    def _build_view(self) -> ft.Control:
        """Create the responsive glove tab layout."""

        content = ft.Column(
            scroll=ft.ScrollMode.HIDDEN,
            spacing=SECTION_GAP,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            controls=[
                self._build_connection_card(),
                self._build_letter_card(),
                self._build_sensor_panel(),
                build_output_card(
                    title="Glove Output",
                    text_control=self.output_text,
                    on_copy=lambda: copy_text(self.page, "\n".join(self._completed_words)),
                    accent=PRIMARY_GREEN,
                    height=150,
                ),
                ft.Row(
                    wrap=True,
                    spacing=10,
                    controls=[
                        ft.Button(
                            "Speak Last Word",
                            icon=ft.Icons.VOLUME_UP,
                            bgcolor=PRIMARY_GREEN,
                            color="#FFFFFF",
                            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
                            on_click=self._speak_last_word,
                        ),
                        ft.OutlinedButton(
                            "Clear",
                            icon=ft.Icons.CLEAR,
                            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
                            on_click=self._clear_output,
                        ),
                        ft.OutlinedButton(
                            "Copy",
                            icon=ft.Icons.CONTENT_COPY,
                            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
                            on_click=lambda _: copy_text(self.page, "\n".join(self._completed_words)),
                        ),
                        self.auto_speak_switch,
                    ],
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

    def _build_connection_card(self) -> ft.Container:
        """Build the connection status row."""

        return ft.Container(
            height=70,
            bgcolor=SURFACE,
            border=border_all(1, BORDER),
            border_radius=CARD_RADIUS,
            padding=14,
            content=ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Row(
                        spacing=9,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[self.status_dot, self.status_text],
                    ),
                    self.connect_button,
                ],
            ),
        )

    def _build_letter_card(self) -> ft.Container:
        """Build the current letter and word display."""

        return ft.Container(
            bgcolor=SURFACE,
            border=border_all(1, BORDER),
            border_radius=CARD_RADIUS,
            padding=20,
            content=ft.Column(
                spacing=6,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    self.current_letter,
                    ft.Text("Current word", size=12, color=TEXT_MUTED, weight=ft.FontWeight.W_600),
                    self.current_word,
                ],
            ),
        )

    def _build_sensor_panel(self) -> ft.Container:
        """Build a collapsible sensor debug panel."""

        labels = [*(f"flex{i}" for i in range(10)), "euler_h", "euler_r", "euler_p", "gyro_x", "gyro_y", "gyro_z"]
        cells: list[ft.Control] = []
        for label in labels:
            value = ft.Text("0.000", size=12, color=TEXT_PRIMARY, weight=ft.FontWeight.W_600)
            self._sensor_values[label] = value
            cells.append(
                ft.Container(
                    col={"xs": 6, "sm": 4, "md": 3},
                    bgcolor=SURFACE_MUTED,
                    border_radius=6,
                    padding=8,
                    content=ft.Column(
                        spacing=2,
                        controls=[
                            ft.Text(label, size=11, color=TEXT_MUTED),
                            value,
                        ],
                    ),
                )
            )

        return ft.Container(
            bgcolor=SURFACE,
            border=border_all(1, BORDER),
            border_radius=CARD_RADIUS,
            content=ft.ExpansionTile(
                title=ft.Text("Sensor data", weight=ft.FontWeight.W_700, color=TEXT_PRIMARY),
                leading=ft.Icon(ft.Icons.SENSORS, color=PRIMARY_GREEN),
                expanded=False,
                controls_padding=12,
                controls=[
                    ft.ResponsiveRow(spacing=8, run_spacing=8, controls=cells, scroll=ft.ScrollMode.HIDDEN),
                ],
            ),
        )

    def _handle_bluetooth_data(self, data: str) -> None:
        """Queue a Bluetooth letter/control packet from a background thread."""

        self._events.put(("data", data))

    def _handle_bluetooth_reading(self, reading: GloveReading) -> None:
        """Queue a Bluetooth sensor reading from a background thread."""

        self._events.put(("reading", reading))

    async def _consume_events(self) -> None:
        """Consume glove events on the Flet event loop."""

        self._consumer_running = True
        try:
            while self._visible or not self._events.empty():
                try:
                    kind, payload = await asyncio.to_thread(self._events.get, True, 0.2)
                except Empty:
                    continue
                if kind == "data":
                    self._process_data(str(payload))
                elif kind == "reading" and isinstance(payload, GloveReading):
                    self._update_sensor_values(payload)
                self._refresh_connection_status()
                self.page.update()
        finally:
            self._consumer_running = False

    async def _watch_silence(self) -> None:
        """Commit a word after two seconds without letter input."""

        self._watcher_running = True
        try:
            while self._visible:
                if self._current_word and time.monotonic() - self._last_letter_at >= 2.0:
                    self._commit_current_word()
                    self.page.update()
                self._refresh_connection_status()
                await asyncio.sleep(0.25)
        finally:
            self._watcher_running = False

    def _process_data(self, data: str) -> None:
        """Apply one glove letter, space, or backspace packet."""

        if data in {" ", "\n", "\r"}:
            self._commit_current_word()
            return
        if data in {"\b", "\x7f"}:
            self._current_word = self._current_word[:-1]
            self.current_word.value = self._current_word
            self.current_letter.value = self._current_word[-1:] or "-"
            return

        letters = "".join(char for char in data.strip().upper() if char.isalpha())
        if not letters:
            return
        now = time.monotonic()
        if self._current_word and now - self._last_letter_at > 2.0:
            self._commit_current_word()
        for letter in letters:
            self._current_word += letter
            self.current_letter.value = letter
            self._last_letter_at = now
        self.current_word.value = self._current_word

    def _commit_current_word(self) -> None:
        """Save, display, and optionally speak the current buffered word."""

        word = self._current_word.strip()
        if not word:
            self._current_word = ""
            self.current_word.value = ""
            self.current_letter.value = "-"
            return
        self._completed_words.append(word)
        self.output_text.value = "\n".join(self._completed_words)
        self.output_text.color = TEXT_PRIMARY
        self.state.record_text(word, "glove")
        if self.state.settings.auto_speak:
            self.state.tts.speak(word)
        self._current_word = ""
        self.current_word.value = ""
        self.current_letter.value = "-"

    def _update_sensor_values(self, reading: GloveReading) -> None:
        """Refresh the compact sensor debug grid."""

        for index, value in enumerate(reading.flex):
            self._sensor_values[f"flex{index}"].value = f"{value:.3f}"
        for name, value in zip(("euler_h", "euler_r", "euler_p"), reading.euler, strict=True):
            self._sensor_values[name].value = f"{value:.3f}"
        for name, value in zip(("gyro_x", "gyro_y", "gyro_z"), reading.gyroscope, strict=True):
            self._sensor_values[name].value = f"{value:.3f}"

    def _refresh_connection_status(self) -> None:
        """Refresh connected/not-connected display state."""

        connected = self.state.bluetooth.is_connected
        self.status_dot.bgcolor = PRIMARY_GREEN if connected else "#DC2626"
        if connected and self.state.bluetooth.connected_device:
            self.status_text.value = self.state.bluetooth.connected_device.name
            self.status_text.color = TEXT_PRIMARY
        else:
            self.status_text.value = "Not connected"
            self.status_text.color = TEXT_MUTED

    def _connect_clicked(self, _: ft.ControlEvent) -> None:
        """Give the user a direct connection hint for Bluetooth or simulator input."""

        self._refresh_connection_status()
        if self.state.bluetooth.is_connected:
            show_snack(self.page, self.state.bluetooth.status_text, bgcolor=PRIMARY_GREEN)
        else:
            show_snack(
                self.page,
                "Use Settings to connect Bluetooth, or run bt_simulator.py for localhost:9999.",
                bgcolor=PRIMARY_BLUE,
            )
        self.page.update()

    def _speak_last_word(self, _: ft.ControlEvent) -> None:
        """Speak the most recently completed glove word."""

        if not self._completed_words:
            show_snack(self.page, "No completed glove word yet.", bgcolor=TEXT_MUTED)
            return
        self.state.tts.speak(self._completed_words[-1])

    def _clear_output(self, _: ft.ControlEvent) -> None:
        """Clear the current and completed glove words."""

        self._current_word = ""
        self._completed_words.clear()
        self.current_word.value = ""
        self.current_letter.value = "-"
        self.output_text.value = "Completed glove words will appear here."
        self.output_text.color = TEXT_MUTED
        self.page.update()

    def _auto_speak_changed(self, event: ft.ControlEvent) -> None:
        """Persist the glove auto-speak setting."""

        self.state.settings.auto_speak = bool(event.control.value)
        try:
            self.state.save_settings()
        except Exception as exc:
            show_snack(self.page, f"Settings could not be saved: {exc}", bgcolor="#B91C1C")


__all__ = ["GloveTab"]
