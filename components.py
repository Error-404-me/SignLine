"""Reusable Flet controls shared by multiple tabs."""

from __future__ import annotations

from collections.abc import Callable

import flet as ft

from theme import (
    BORDER,
    CARD_RADIUS,
    PRIMARY_BLUE,
    SURFACE,
    SURFACE_MUTED,
    TEXT_MUTED,
    TEXT_PRIMARY,
    border_all,
)


def show_snack(page: ft.Page, message: str, bgcolor: str | None = None) -> None:
    """Show a short snack-bar message."""

    page.show_dialog(
        ft.SnackBar(
            content=ft.Text(message, color="#FFFFFF"),
            bgcolor=bgcolor or TEXT_PRIMARY,
            show_close_icon=True,
        )
    )


def copy_text(page: ft.Page, text: str) -> None:
    """Copy text to the system clipboard and notify the user."""

    if not text.strip():
        show_snack(page, "Nothing to copy yet.", bgcolor=TEXT_MUTED)
        return
    page.clipboard.set(text)
    show_snack(page, "Copied to clipboard.", bgcolor=PRIMARY_BLUE)


def build_section_title(title: str, subtitle: str | None = None) -> ft.Column:
    """Build a compact section heading."""

    controls: list[ft.Control] = [
        ft.Text(title, size=16, weight=ft.FontWeight.W_700, color=TEXT_PRIMARY)
    ]
    if subtitle:
        controls.append(ft.Text(subtitle, size=12, color=TEXT_MUTED))
    return ft.Column(controls=controls, spacing=2)


def build_output_card(
    *,
    title: str,
    text_control: ft.Text,
    on_copy: Callable[[], None],
    accent: str,
    on_clear: Callable[[], None] | None = None,
    extra_actions: list[ft.Control] | None = None,
    height: int = 188,
) -> ft.Container:
    """Build the scrollable text output card used by speech-style tabs."""

    action_buttons: list[ft.Control] = []
    if on_clear is not None:
        action_buttons.append(
            ft.IconButton(
                icon=ft.Icons.DELETE_OUTLINE,
                icon_color=TEXT_MUTED,
                tooltip="Clear output",
                on_click=lambda _: on_clear(),
            )
        )
    if extra_actions:
        action_buttons.extend(extra_actions)
    action_buttons.append(
        ft.IconButton(
            icon=ft.Icons.CONTENT_COPY,
            icon_color=accent,
            tooltip="Copy output",
            on_click=lambda _: on_copy(),
        )
    )

    return ft.Container(
        bgcolor=SURFACE,
        border=border_all(1, BORDER),
        border_radius=CARD_RADIUS,
        padding=14,
        content=ft.Column(
            spacing=10,
            controls=[
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Text(
                            title,
                            size=15,
                            weight=ft.FontWeight.W_700,
                            color=TEXT_PRIMARY,
                        ),
                        ft.Row(
                            spacing=0,
                            tight=True,
                            controls=action_buttons,
                        ),
                    ],
                ),
                ft.Container(
                    height=height,
                    bgcolor=SURFACE_MUTED,
                    border_radius=6,
                    padding=12,
                    content=ft.Column(
                        width=float('inf'),
                        controls=[text_control],
                        scroll=ft.ScrollMode.AUTO,
                        spacing=4,
                    ),
                ),
            ],
        ),
    )


def build_badge(text: str, color: str, icon: int | None = None) -> ft.Container:
    """Build a small rounded status badge."""

    controls: list[ft.Control] = []
    if icon is not None:
        controls.append(ft.Icon(icon, size=14, color=color))
    controls.append(ft.Text(text, size=12, weight=ft.FontWeight.W_600, color=color))
    return ft.Container(
        bgcolor="#FFFFFF",
        border=border_all(1, color),
        border_radius=999,
        padding=ft.Padding(10, 5, 10, 5),
        content=ft.Row(spacing=5, tight=True, controls=controls),
    )


__all__ = [
    "build_badge",
    "build_output_card",
    "build_section_title",
    "copy_text",
    "show_snack",
]
