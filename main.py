"""Flet entry point for Speak & Sign to Text."""


from __future__ import annotations

import asyncio
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
import sys

import flet as ft

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app_bar import build_app_bar
from app_state import AppState, load_app_settings
from services.bluetooth_service import BluetoothManager, BluetoothMode
from services.db_service import HistoryDatabase
from services.tts_service import TTSService
from splash_screen import build_splash_screen
from tabs.glove_tab import GloveTab
from tabs.history_tab import HistoryTab
from tabs.settings_tab import SettingsTab
from tabs.speech_tab import SpeechTab
from theme import BACKGROUND, PRIMARY_BLUE, SURFACE, TAB_TITLES, build_theme, tab_accent, tab_tint


INFO_TEXT = {
    0: "Use the Start Listening button to convert live audio chunks to text using the selected local Vosk or Whisper model.",
    1: (
        "Connect your ESP32 BNO055 glove via Bluetooth. Flex and motion sensors "
        "classify your hand signs into letters that build into words, which can be spoken aloud."
    ),
    2: "Speech and glove words are saved locally in SQLite and can be searched, copied, or cleared.",
    3: "Choose offline speech models and connect an Android Bluetooth SPP device for text output.",
}


def main(page: ft.Page) -> None:
    """Build and run the Flet application."""

    page.title = "Speak & Sign to Text"
    page.theme = build_theme()
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor = BACKGROUND
    page.padding = 0
    page.spacing = 0
    try:
        page.window.width = 410
        page.window.height = 820
        page.window.min_width = 360
        page.window.min_height = 720
        page.window.alignment = ft.Alignment.CENTER
    except Exception:
        pass

    assets_dir = APP_DIR / "assets"
    data_dir = APP_DIR / "data"
    settings = load_app_settings(data_dir / "settings.json")
    bluetooth = BluetoothManager(assets_dir=assets_dir)
    bluetooth.set_mode(BluetoothMode(settings.bluetooth_mode))
    state = AppState(
        assets_dir=assets_dir,
        data_dir=data_dir,
        settings=settings,
        db=HistoryDatabase(data_dir / "history.db"),
        bluetooth=bluetooth,
        tts=TTSService(),
        settings_path=data_dir / "settings.json",
    )

    tabs = [
        SpeechTab(page, state),
        GloveTab(page, state),
        HistoryTab(page, state),
        SettingsTab(page, state),
    ]
    current = {"index": 0}
    content_host = ft.Container(expand=True, content=build_splash_screen())

    def navigate(index: int) -> None:
        """Switch to a tab and update shared chrome."""

        if index == current["index"]:
            return
        tabs[current["index"]].on_hidden()
        current["index"] = index
        navigation_bar.selected_index = index
        navigation_bar.indicator_color = tab_tint(index)
        content_host.content = tabs[index].build()
        page.appbar = build_app_bar(
            title=TAB_TITLES[index],
            accent=tab_accent(index),
            on_menu=open_drawer,
            on_info=open_info,
        )
        if page.drawer is not None:
            page.drawer.selected_index = index
        tabs[index].on_visible()
        page.update()

    def navigation_changed(event: ft.ControlEvent) -> None:
        """Handle bottom navigation changes."""

        navigate(int(event.control.selected_index))

    navigation_bar = ft.NavigationBar(
        selected_index=0,
        bgcolor=SURFACE,
        indicator_color=tab_tint(0),
        elevation=0,
        destinations=[
            ft.NavigationBarDestination(
                icon=ft.Icons.MIC_OUTLINED,
                selected_icon=ft.Icons.MIC,
                label="Speech",
            ),
            ft.NavigationBarDestination(
                icon=ft.Icons.FRONT_HAND_OUTLINED,
                selected_icon=ft.Icons.FRONT_HAND,
                label="Glove",
            ),
            ft.NavigationBarDestination(
                icon=ft.Icons.HISTORY_OUTLINED,
                selected_icon=ft.Icons.HISTORY,
                label="History",
            ),
            ft.NavigationBarDestination(
                icon=ft.Icons.SETTINGS_OUTLINED,
                selected_icon=ft.Icons.SETTINGS,
                label="Settings",
            ),
        ],
        on_change=navigation_changed,
    )

    def drawer_changed(event: ft.ControlEvent) -> None:
        """Handle hamburger drawer tab selection."""

        if event.control.selected_index is None:
            return
        selected = int(event.control.selected_index)
        if 0 <= selected <= 3:
            if page.drawer is not None:
                page.run_task(page.close_drawer)
            navigate(selected)

    page.drawer = ft.NavigationDrawer(
        selected_index=0,
        indicator_color=tab_tint(0),
        on_change=drawer_changed,
        controls=[
            ft.NavigationDrawerDestination(
                label="Speech to Text",
                icon=ft.Icons.MIC_OUTLINED,
                selected_icon=ft.Icons.MIC,
            ),
            ft.NavigationDrawerDestination(
                label="Glove Sign",
                icon=ft.Icons.FRONT_HAND_OUTLINED,
                selected_icon=ft.Icons.FRONT_HAND,
            ),
            ft.NavigationDrawerDestination(
                label="History",
                icon=ft.Icons.HISTORY_OUTLINED,
                selected_icon=ft.Icons.HISTORY,
            ),
            ft.NavigationDrawerDestination(
                label="Settings",
                icon=ft.Icons.SETTINGS_OUTLINED,
                selected_icon=ft.Icons.SETTINGS,
            ),
        ],
    )

    async def show_app_shell() -> None:
        """Reveal the main app after the launch splash."""

        await asyncio.sleep(1.15)
        content_host.content = tabs[0].build()
        page.appbar = build_app_bar(
            title=TAB_TITLES[0],
            accent=PRIMARY_BLUE,
            on_menu=open_drawer,
            on_info=open_info,
        )
        page.navigation_bar = navigation_bar
        page.update()

    def open_drawer() -> None:
        """Open the hamburger navigation drawer."""

        if page.drawer is not None:
            page.run_task(page.show_drawer)

    def open_info() -> None:
        """Show a short information dialog for the active tab."""

        index = current["index"]
        dialog = ft.AlertDialog(
            modal=False,
            title=ft.Text(TAB_TITLES[index]),
            content=ft.Text(INFO_TEXT[index]),
            actions_alignment=ft.MainAxisAlignment.END,
        )
        dialog.actions = [
            ft.TextButton("OK", on_click=lambda _: close_dialog(dialog)),
        ]
        page.show_dialog(dialog)

    def close_dialog(dialog: ft.AlertDialog) -> None:
        """Close an app dialog."""

        if page.pop_dialog() is None:
            dialog.open = False
        page.update()

    def cleanup(_: ft.ControlEvent) -> None:
        """Release device resources when the app is closed."""

        for tab in tabs:
            tab.on_hidden()
        state.tts.stop()
        state.bluetooth.disconnect()
        state.db.close()

    page.on_close = cleanup
    page.add(content_host)
    page.run_task(show_app_shell)


if __name__ == "__main__":
    ft.run(main, assets_dir=str(APP_DIR / "assets"))


__all__ = ["main"]
