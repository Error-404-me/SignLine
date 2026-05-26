"""Shared AppBar builder."""

from __future__ import annotations

from collections.abc import Callable

import flet as ft

from theme import SURFACE, TEXT_PRIMARY


def build_app_bar(
    *,
    title: str,
    accent: str,
    on_menu: Callable[[], None],
    on_info: Callable[[], None],
) -> ft.AppBar:
    """Create the global top AppBar with menu, dynamic title, and info action."""

    return ft.AppBar(
        leading=ft.IconButton(
            icon=ft.Icons.MENU,
            icon_color=accent,
            tooltip="Menu",
            on_click=lambda _: on_menu(),
        ),
        title=ft.Text(title, weight=ft.FontWeight.W_700, color=TEXT_PRIMARY),
        center_title=False,
        bgcolor=SURFACE,
        elevation=0,
        actions=[
            ft.IconButton(
                icon=ft.Icons.INFO_OUTLINED,
                icon_color=accent,
                tooltip="About this tab",
                on_click=lambda _: on_info(),
            )
        ],
    )


__all__ = ["build_app_bar"]
