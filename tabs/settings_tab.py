"""Settings and Bluetooth tab."""

from __future__ import annotations

import threading

import flet as ft
from flet_permission_handler import PermissionHandler

from app_state import AppState
from components import build_badge, build_section_title, show_snack
from services.bluetooth_service import BluetoothDeviceInfo, BluetoothError, BluetoothMode
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
from ui_permissions import ensure_bluetooth_permissions


class SettingsTab:
    """Flet settings view including speech and Android Bluetooth controls."""

    def __init__(self, page: ft.Page, state: AppState) -> None:
        """Create settings controls."""

        self.page = page
        self.state = state
        self.permission_handler = PermissionHandler()
        self._register_permission_handler()
        self.speech_engine_dropdown = ft.Dropdown(
            label="Speech engine",
            value=self.state.settings.speech_engine,
            options=[
                ft.DropdownOption(key=key, text=value)
                for key, value in self.state.settings.available_speech_engines.items()
            ],
            on_select=self._speech_engine_changed,
            expand=True,
        )
        self.speech_model_dropdown = ft.Dropdown(
            label="Whisper model",
            value=self.state.settings.speech_model,
            options=[
                ft.DropdownOption(key=key, text=value)
                for key, value in self.state.settings.available_speech_models.items()
            ],
            on_select=self._speech_model_changed,
            expand=True,
        )
        self.speech_language_dropdown = ft.Dropdown(
            label="Speech language",
            value=self.state.settings.speech_language,
            options=[
                ft.DropdownOption(key=key, text=value)
                for key, value in self.state.settings.available_speech_languages.items()
            ],
            on_select=self._speech_language_changed,
            expand=True,
        )
        self.speech_quality_note = ft.Text(size=12, color=TEXT_MUTED)
        self.bluetooth_status = build_badge(
            self.state.bluetooth.status_text,
            PRIMARY_GREEN if self.state.bluetooth.is_connected else TEXT_MUTED,
            icon=ft.Icons.BLUETOOTH_CONNECTED
            if self.state.bluetooth.is_connected
            else ft.Icons.BLUETOOTH,
        )
        # self.bluetooth_note = ft.Text(
        #     "Bluetooth scan and RFCOMM output are enabled on Android builds. Desktop runs in mock mode.",
        #     size=12,
        #     color=TEXT_MUTED,
        # )
        self.bluetooth_mode_selector = ft.SegmentedButton(
            selected=[self.state.bluetooth.mode.value],
            segments=[
                ft.Segment(
                    value=BluetoothMode.LETTER.value,
                    icon=ft.Icon(ft.Icons.TEXT_FIELDS),
                    label=ft.Text("Letter"),
                ),
                ft.Segment(
                    value=BluetoothMode.SENSOR.value,
                    icon=ft.Icon(ft.Icons.SENSORS),
                    label=ft.Text("Sensor"),
                ),
            ],
            show_selected_icon=False,
            on_change=self._bluetooth_mode_changed,
        )
        self.auto_speak_switch = ft.Switch(
            value=self.state.settings.auto_speak,
            active_color=PRIMARY_GREEN,
            on_change=self._auto_speak_changed,
        )
        self.mic_gain_value = ft.Text(
            f"{self.state.settings.mic_gain:.1f}\u00d7",
            width=60,
            text_align=ft.TextAlign.RIGHT,
            color=TEXT_MUTED,
        )
        self.mic_gain_slider = ft.Slider(
            min=0.5,
            max=3.0,
            divisions=25,
            value=self.state.settings.mic_gain,
            expand=True,
            on_change=self._mic_gain_changed,
        )
        self.highpass_switch = ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Text(value="Remove low-frequency rumble"),
                ft.Switch(
                    value=self.state.settings.highpass_enabled,
                    active_color=PRIMARY_BLUE,
                    on_change=self._highpass_enabled_changed,
                ),
            ]
        )
        self.highpass_cutoff_value = ft.Text(
            f"{self.state.settings.highpass_cutoff_hz:.0f} Hz",
            width=60,
            text_align=ft.TextAlign.RIGHT,
            color=TEXT_MUTED,
        )
        self.highpass_cutoff_slider = ft.Slider(
            min=60,
            max=400,
            divisions=34,
            value=self.state.settings.highpass_cutoff_hz,
            expand=True,
            disabled=not self.state.settings.highpass_enabled,
            on_change=self._highpass_cutoff_changed,
        )
        self.denoise_switch = ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            controls=[
                ft.Text(value="Spectral noise reduction"),
                ft.Switch(
                    value=self.state.settings.denoise_enabled,
                    active_color=PRIMARY_BLUE,
                    on_change=self._denoise_enabled_changed,
                ),
            ]
        )
        self.denoise_strength_value = ft.Text(
            f"{self.state.settings.denoise_prop_decrease:.0%}",
            width=60,
            text_align=ft.TextAlign.RIGHT,
            color=TEXT_MUTED,
        )
        self.denoise_strength_slider = ft.Slider(
            min=0.1,
            max=1.0,
            divisions=9,
            value=self.state.settings.denoise_prop_decrease,
            expand=True,
            disabled=not self.state.settings.denoise_enabled,
            on_change=self._denoise_strength_changed,
        )
        self.denoise_stationary_switch = ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            controls=[
                ft.Text(value="Stationary noise (fan, AC hum)"),
                ft.Switch(
                    value=self.state.settings.denoise_stationary,
                    active_color=PRIMARY_BLUE,
                    disabled=not self.state.settings.denoise_enabled,
                    on_change=self._denoise_stationary_changed,
                ),
            ]
        )
        self.no_speech_threshold_value = ft.Text(
            f"{self.state.settings.no_speech_threshold:.2f}",
            width=60,
            text_align=ft.TextAlign.RIGHT,
            color=TEXT_MUTED,
        )
        self.no_speech_threshold_slider = ft.Slider(
            min=0.3,
            max=0.95,
            divisions=13,
            value=self.state.settings.no_speech_threshold,
            expand=True,
            on_change=self._no_speech_threshold_changed,
        )
        self.vad_silence_value = ft.Text(
            f"{self.state.settings.vad_silence_ms:.0f} ms",
            width=60,
            text_align=ft.TextAlign.RIGHT,
            color=TEXT_MUTED,
        )
        self.vad_silence_slider = ft.Slider(
            min=200,
            max=2000,
            divisions=18,
            value=self.state.settings.vad_silence_ms,
            expand=True,
            on_change=self._vad_silence_changed,
        )
        self.vad_threshold_value = ft.Text(
            f"{self.state.settings.vad_threshold:.1f}",
            width=60,
            text_align=ft.TextAlign.RIGHT,
            color=TEXT_MUTED,
        )
        self.vad_threshold_slider = ft.Slider(
            min=0.1,
            max=0.9,
            divisions=8,
            value=self.state.settings.vad_threshold,
            expand=True,
            on_change=self._vad_threshold_changed,
        )
        self.device_list = ft.ListView(height=210, spacing=8)
        self.scan_button = ft.Button(
            content=ft.Text("Scan for devices", weight=ft.FontWeight.W_700),
            icon=ft.Icons.BLUETOOTH_SEARCHING,
            bgcolor=PRIMARY_BLUE,
            color="#FFFFFF",
            height=44,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
            on_click=self._scan_clicked,
        )
        self.disconnect_button = ft.OutlinedButton(
            content=ft.Text("Disconnect"),
            icon=ft.Icons.BLUETOOTH_DISABLED,
            height=44,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
            on_click=self._disconnect_clicked,
            disabled=not self.state.bluetooth.is_connected,
        )
        self._view = self._build_view()
        self.refresh_bluetooth_status()
        self._refresh_model_notes()

    def _register_permission_handler(self) -> None:
        """Register the shared permission handler service with the page."""

        try:
            self.page.register_service(self.permission_handler)
        except Exception:
            pass

    def build(self) -> ft.Control:
        """Return the settings root control."""

        return self._view

    def on_visible(self) -> None:
        """Refresh Bluetooth and speech setting notes when the tab becomes active."""

        self.refresh_bluetooth_status()
        self.bluetooth_mode_selector.selected = [self.state.bluetooth.mode.value]
        self.auto_speak_switch.value = self.state.settings.auto_speak
        self._sync_audio_controls()
        self._refresh_model_notes()
        self.page.update()

    def on_hidden(self) -> None:
        """Leave current settings intact when navigating away."""

    def refresh_bluetooth_status(self) -> None:
        """Update Bluetooth status badge and buttons."""

        connected = self.state.bluetooth.is_connected
        self.bluetooth_status.content.controls[-1].value = self.state.bluetooth.status_text
        self.bluetooth_status.border = border_all(1, PRIMARY_GREEN if connected else TEXT_MUTED)
        self.bluetooth_status.content.controls[0].name = (
            ft.Icons.BLUETOOTH_CONNECTED if connected else ft.Icons.BLUETOOTH
        )
        self.bluetooth_status.content.controls[0].color = PRIMARY_GREEN if connected else TEXT_MUTED
        self.bluetooth_status.content.controls[-1].color = PRIMARY_GREEN if connected else TEXT_MUTED
        self.disconnect_button.disabled = not connected

    def _build_view(self) -> ft.Control:
        """Create the responsive settings layout."""

        sections = ft.ResponsiveRow(
            run_spacing=SECTION_GAP,
            spacing=SECTION_GAP,
            controls=[
                ft.Container(
                    col={"xs": 12, "md": 6},
                    content=self._build_speech_card(),
                ),
                ft.Container(
                    col={"xs": 12},
                    content=self._build_bluetooth_card(),
                ),
            ],
        )
        return ft.SafeArea(
            content=ft.Container(
                expand=True,
                padding=PAGE_PADDING,
                content=ft.Column(
                    expand=True,
                    scroll=ft.ScrollMode.AUTO,
                    controls=[sections],
                ),
            )
        )

    def _build_speech_card(self) -> ft.Container:
        """Build the speech settings card."""

        return ft.Container(
            bgcolor=SURFACE,
            border=border_all(1, BORDER),
            border_radius=CARD_RADIUS,
            padding=16,
            content=ft.Column(
                spacing=14,
                controls=[
                    build_section_title("Speech Settings", "Offline speech recognition"),
                    self.speech_engine_dropdown,
                    self.speech_language_dropdown,
                    self.speech_model_dropdown,
                    self.speech_quality_note,
                    self._build_audio_filters_section(),
                    self._build_whisper_vad_section(),
                ],
            ),
        )

    def _build_audio_filters_section(self) -> ft.ExpansionTile:
        """Build audio preprocessing controls."""

        return ft.ExpansionTile(
            title=ft.Text("Audio Filters", weight=ft.FontWeight.W_700, color=TEXT_PRIMARY),
            leading=ft.Icon(ft.Icons.TUNE, color=PRIMARY_BLUE),
            expanded=True,
            controls_padding=12,
            expanded_cross_axis_alignment=ft.CrossAxisAlignment.STRETCH,
            controls=[
                self._slider_column(label="Mic Gain", slider=self.mic_gain_slider, value_text=self.mic_gain_value),
                self.highpass_switch,
                self._slider_column(
                    "High-pass Cutoff",
                    self.highpass_cutoff_slider,
                    self.highpass_cutoff_value,
                ),
                self.denoise_switch,
                self._slider_column(
                    "Noise Reduction Strength",
                    self.denoise_strength_slider,
                    self.denoise_strength_value,
                ),
                self.denoise_stationary_switch,
            ],
        )

    def _build_whisper_vad_section(self) -> ft.ExpansionTile:
        """Build Whisper and voice activity detection controls."""

        return ft.ExpansionTile(
            title=ft.Text("Whisper / VAD", weight=ft.FontWeight.W_700, color=TEXT_PRIMARY),
            leading=ft.Icon(ft.Icons.GRAPHIC_EQ, color=PRIMARY_BLUE),
            expanded=False,
            controls_padding=12,
            expanded_cross_axis_alignment=ft.CrossAxisAlignment.STRETCH,
            controls=[
                self._slider_column(
                    "No-speech Threshold",
                    self.no_speech_threshold_slider,
                    self.no_speech_threshold_value,
                ),
                self._slider_column(
                    "VAD Silence Duration",
                    self.vad_silence_slider,
                    self.vad_silence_value,
                ),
                self._slider_column(
                    "VAD Sensitivity",
                    self.vad_threshold_slider,
                    self.vad_threshold_value,
                ),
            ],
        )

    def _slider_column(
        self,
        label: str,
        slider: ft.Slider,
        value_text: ft.Text,
    ) -> ft.Row:
        """Build one settings slider row."""

        return ft.Column(
            spacing=0,
            margin=ft.Margin.only(bottom=12),
            controls=[
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Text(label, width=160, color=TEXT_PRIMARY),
                        value_text,
                    ]
                ),
                slider,
            ],
        )

    def _build_bluetooth_card(self) -> ft.Container:
        """Build the Bluetooth settings panel."""

        return ft.Container(
            bgcolor=SURFACE,
            border=border_all(1, BORDER),
            border_radius=CARD_RADIUS,
            padding=16,
            content=ft.Column(
                spacing=14,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            build_section_title("Bluetooth", "Send recognised text over SPP"),
                            self.bluetooth_status,
                        ],
                    ),
                    #self.bluetooth_note,
                    ft.Text("Incoming glove mode", weight=ft.FontWeight.W_600, color=TEXT_PRIMARY),
                    self.bluetooth_mode_selector,
                    ft.Row(
                        vertical_alignment=ft.CrossAxisAlignment.CENTER, 
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        controls=[
                            ft.Text(value="Auto-speak glove words"),
                            self.auto_speak_switch,
                        ]    
                    ),
                    ft.Row(controls=[self.scan_button, self.disconnect_button], wrap=True, spacing=10),
                    ft.Container(
                        bgcolor=SURFACE_MUTED,
                        border_radius=8,
                        padding=8,
                        content=self.device_list,
                    ),
                ],
            ),
        )

    def _speech_engine_changed(self, event: ft.ControlEvent) -> None:
        """Store selected speech recognition engine."""

        self.state.settings.speech_engine = str(event.control.value)
        self._save_settings()
        self._refresh_model_notes()
        show_snack(self.page, "Speech engine updated.", bgcolor=PRIMARY_BLUE)
        self.page.update()

    def _speech_model_changed(self, event: ft.ControlEvent) -> None:
        """Store selected speech model."""

        self.state.settings.speech_model = str(event.control.value)
        self._save_settings()
        self._refresh_model_notes()
        show_snack(self.page, "Speech model updated.", bgcolor=PRIMARY_BLUE)
        self.page.update()

    def _speech_language_changed(self, event: ft.ControlEvent) -> None:
        """Store selected speech recognition language."""

        self.state.settings.speech_language = str(event.control.value)
        self._save_settings()
        self._refresh_model_notes()
        show_snack(self.page, "Speech language updated.", bgcolor=PRIMARY_BLUE)
        self.page.update()

    def _save_settings(self) -> None:
        """Persist selected settings and surface storage failures."""

        try:
            self.state.save_settings()
        except Exception as exc:
            show_snack(self.page, f"Settings could not be saved: {exc}", bgcolor="#B91C1C")

    def _bluetooth_mode_changed(self, event: ft.ControlEvent) -> None:
        """Store selected glove Bluetooth input mode."""

        selected = list(event.control.selected)
        if not selected:
            event.control.selected = [self.state.bluetooth.mode.value]
            self.page.update()
            return
        mode = BluetoothMode(selected[0])
        self.state.bluetooth.set_mode(mode)
        self.state.settings.bluetooth_mode = mode.value
        self._save_settings()
        show_snack(self.page, f"Bluetooth mode: {mode.value.title()}", bgcolor=PRIMARY_GREEN)
        self.page.update()

    def _auto_speak_changed(self, event: ft.ControlEvent) -> None:
        """Store auto-speak preference."""

        self.state.settings.auto_speak = bool(event.control.value)
        self._save_settings()
        self.page.update()

    def _mic_gain_changed(self, event: ft.ControlEvent) -> None:
        """Store microphone gain."""

        value = float(event.control.value)
        self.state.settings.mic_gain = value
        self.mic_gain_value.value = f"{value:.1f}\u00d7"
        self._save_settings()
        self.page.update()

    def _highpass_enabled_changed(self, event: ft.ControlEvent) -> None:
        """Enable or disable high-pass filtering."""

        enabled = bool(event.control.value)
        self.state.settings.highpass_enabled = enabled
        self.highpass_cutoff_slider.disabled = not enabled
        self._save_settings()
        self.page.update()

    def _highpass_cutoff_changed(self, event: ft.ControlEvent) -> None:
        """Store high-pass cutoff frequency."""

        value = float(event.control.value)
        self.state.settings.highpass_cutoff_hz = value
        self.highpass_cutoff_value.value = f"{value:.0f} Hz"
        self._save_settings()
        self.page.update()

    def _denoise_enabled_changed(self, event: ft.ControlEvent) -> None:
        """Enable or disable spectral noise reduction."""

        enabled = bool(event.control.value)
        self.state.settings.denoise_enabled = enabled
        self.denoise_strength_slider.disabled = not enabled
        self.denoise_stationary_switch.disabled = not enabled
        self._save_settings()
        self.page.update()

    def _denoise_strength_changed(self, event: ft.ControlEvent) -> None:
        """Store spectral noise reduction strength."""

        value = float(event.control.value)
        self.state.settings.denoise_prop_decrease = value
        self.denoise_strength_value.value = f"{value:.0%}"
        self._save_settings()
        self.page.update()

    def _denoise_stationary_changed(self, event: ft.ControlEvent) -> None:
        """Store stationary noise preference."""

        self.state.settings.denoise_stationary = bool(event.control.value)
        self._save_settings()
        self.page.update()

    def _no_speech_threshold_changed(self, event: ft.ControlEvent) -> None:
        """Store Whisper no-speech threshold."""

        value = float(event.control.value)
        self.state.settings.no_speech_threshold = value
        self.no_speech_threshold_value.value = f"{value:.2f}"
        self._save_settings()
        self.page.update()

    def _vad_silence_changed(self, event: ft.ControlEvent) -> None:
        """Store VAD silence duration in milliseconds."""

        value = int(round(float(event.control.value)))
        self.state.settings.vad_silence_ms = value
        self.vad_silence_value.value = f"{value:.0f} ms"
        self._save_settings()
        self.page.update()

    def _vad_threshold_changed(self, event: ft.ControlEvent) -> None:
        """Store VAD sensitivity threshold."""

        value = float(event.control.value)
        self.state.settings.vad_threshold = value
        self.vad_threshold_value.value = f"{value:.1f}"
        self._save_settings()
        self.page.update()

    def _sync_audio_controls(self) -> None:
        """Refresh audio filter controls from current settings."""

        settings = self.state.settings
        self.mic_gain_slider.value = settings.mic_gain
        self.mic_gain_value.value = f"{settings.mic_gain:.1f}\u00d7"
        self.highpass_switch.value = settings.highpass_enabled
        self.highpass_cutoff_slider.value = settings.highpass_cutoff_hz
        self.highpass_cutoff_slider.disabled = not settings.highpass_enabled
        self.highpass_cutoff_value.value = f"{settings.highpass_cutoff_hz:.0f} Hz"
        self.denoise_switch.value = settings.denoise_enabled
        self.denoise_strength_slider.value = settings.denoise_prop_decrease
        self.denoise_strength_slider.disabled = not settings.denoise_enabled
        self.denoise_strength_value.value = f"{settings.denoise_prop_decrease:.0%}"
        self.denoise_stationary_switch.value = settings.denoise_stationary
        self.denoise_stationary_switch.disabled = not settings.denoise_enabled
        self.no_speech_threshold_slider.value = settings.no_speech_threshold
        self.no_speech_threshold_value.value = f"{settings.no_speech_threshold:.2f}"
        self.vad_silence_slider.value = settings.vad_silence_ms
        self.vad_silence_value.value = f"{settings.vad_silence_ms:.0f} ms"
        self.vad_threshold_slider.value = settings.vad_threshold
        self.vad_threshold_value.value = f"{settings.vad_threshold:.1f}"

    def _refresh_model_notes(self) -> None:
        """Refresh speech model quality and availability notes."""

        uses_vosk = self.state.settings.speech_engine == "vosk"
        self.speech_model_dropdown.disabled = uses_vosk
        if uses_vosk:
            vosk_path = self.state.settings.vosk_model_path(self.state.assets_dir)
            if vosk_path.exists():
                self.speech_quality_note.value = (
                    f"Vosk uses {vosk_path.name} and emits finalized speech "
                    "phrases instead of early partial guesses."
                )
                self.speech_quality_note.color = TEXT_MUTED
            else:
                self.speech_quality_note.value = (
                    f"Missing Vosk model folder: {vosk_path}. Add the model "
                    "before starting live listening."
                )
                self.speech_quality_note.color = "#B45309"
        elif self.state.settings.speech_language == "tl":
            self.speech_quality_note.value = (
                "Whisper Filipino accuracy is best with the base model, clear "
                "audio, and longer phrases. Tiny is faster but less accurate."
            )
            self.speech_quality_note.color = TEXT_MUTED
        else:
            self.speech_quality_note.value = "Whisper English uses the selected local model."
            self.speech_quality_note.color = TEXT_MUTED

    def _scan_clicked(self, _: ft.ControlEvent) -> None:
        """Start the async permission gate before scanning devices."""

        self.page.run_task(self._scan_after_permissions)

    async def _scan_after_permissions(self) -> None:
        """Request Bluetooth permissions on the UI task, then scan in a worker."""

        if not await ensure_bluetooth_permissions(self.page, self.permission_handler):
            return

        self.scan_button.disabled = True
        self.scan_button.content = ft.Text("Scanning...", weight=ft.FontWeight.W_700)
        self.page.update()

        def worker() -> None:
            try:
                devices = self.state.bluetooth.scan_devices()
                self.page.run_task(self._show_devices, devices)
            except BluetoothError as exc:
                self.page.run_task(self._show_bluetooth_error, str(exc))

        threading.Thread(target=worker, name="bluetooth-scan", daemon=True).start()

    async def _show_devices(self, devices: list[BluetoothDeviceInfo]) -> None:
        """Render discovered or paired devices."""

        self.scan_button.disabled = False
        self.scan_button.content = ft.Text("Scan for devices", weight=ft.FontWeight.W_700)
        if not devices:
            self.device_list.controls = [
                ft.Container(
                    padding=12,
                    content=ft.Text("No paired Bluetooth devices found.", color=TEXT_MUTED),
                )
            ]
        else:
            self.device_list.controls = [self._device_tile(device) for device in devices]
        self.page.update()

    async def _show_bluetooth_error(self, message: str) -> None:
        """Display a Bluetooth error and restore scan controls."""

        self.scan_button.disabled = False
        self.scan_button.content = ft.Text("Scan for devices", weight=ft.FontWeight.W_700)
        show_snack(self.page, message, bgcolor="#B91C1C")
        self.page.update()

    def _device_tile(self, device: BluetoothDeviceInfo) -> ft.Control:
        """Build one Bluetooth device row."""

        return ft.Container(
            bgcolor="#FFFFFF",
            border=border_all(1, BORDER),
            border_radius=8,
            content=ft.ListTile(
                leading=ft.Icon(ft.Icons.BLUETOOTH, color=PRIMARY_BLUE),
                title=ft.Text(device.name, weight=ft.FontWeight.W_600, color=TEXT_PRIMARY),
                subtitle=ft.Text(device.address, color=TEXT_MUTED),
                trailing=ft.Button(
                    "Connect",
                    bgcolor=PRIMARY_GREEN,
                    color="#FFFFFF",
                    style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
                    on_click=lambda _, info=device: self._connect_clicked(info),
                ),
            ),
        )

    def _connect_clicked(self, device: BluetoothDeviceInfo) -> None:
        """Start the async permission gate before connecting."""

        self.page.run_task(self._connect_after_permissions, device)

    async def _connect_after_permissions(self, device: BluetoothDeviceInfo) -> None:
        """Request Bluetooth permissions on the UI task, then connect in a worker."""

        if not await ensure_bluetooth_permissions(self.page, self.permission_handler):
            return

        show_snack(self.page, f"Connecting to {device.name}...", bgcolor=PRIMARY_BLUE)

        def worker() -> None:
            try:
                self.state.bluetooth.connect(device.address, device.name)
                self.page.run_task(self._connected, device)
            except BluetoothError as exc:
                self.page.run_task(self._show_bluetooth_error, str(exc))

        threading.Thread(target=worker, name="bluetooth-connect", daemon=True).start()

    async def _connected(self, device: BluetoothDeviceInfo) -> None:
        """Update UI after a Bluetooth connection succeeds."""

        self.refresh_bluetooth_status()
        if self.state.bluetooth.is_connected:
            message = f"Connected to {device.name}."
            color = PRIMARY_GREEN
        else:
            message = self.state.bluetooth.status_text
            color = PRIMARY_BLUE
        show_snack(self.page, message, bgcolor=color)
        self.page.update()

    def _disconnect_clicked(self, _: ft.ControlEvent) -> None:
        """Disconnect the active Bluetooth socket."""

        self.state.bluetooth.disconnect()
        self.refresh_bluetooth_status()
        show_snack(self.page, "Bluetooth disconnected.", bgcolor=PRIMARY_BLUE)
        self.page.update()


__all__ = ["SettingsTab"]
