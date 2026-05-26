"""Splash screen control shown while the app shell is prepared."""

from __future__ import annotations

import flet as ft

from theme import BACKGROUND, CENTER_ALIGNMENT, PRIMARY_BLUE, SURFACE, TEXT_MUTED, TEXT_PRIMARY


def build_splash_screen() -> ft.Control:
    """Build the launch splash screen."""

    return ft.Container(
        expand=True,
        bgcolor=BACKGROUND,
        alignment=CENTER_ALIGNMENT,
        content=ft.Column(
            width=320,
            spacing=18,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            controls=[
                ft.Container(
                    width=112,
                    height=112,
                    bgcolor=SURFACE,
                    border_radius=28,
                    shadow=ft.BoxShadow(
                        blur_radius=24,
                        spread_radius=1,
                        color="#1E40AF22",
                        offset=ft.Offset(0, 10),
                    ),
                    alignment=CENTER_ALIGNMENT,
                    content=ft.Image(
                        src="icon.png",
                        width=72,
                        height=72,
                        fit=ft.BoxFit.CONTAIN,
                    ),
                ),
                ft.Column(
                    spacing=5,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Text(
                            "Speak & Sign to Text",
                            size=24,
                            weight=ft.FontWeight.W_800,
                            color=TEXT_PRIMARY,
                            text_align=ft.TextAlign.CENTER,
                        ),
                        ft.Text(
                            "Offline speech and glove text",
                            size=14,
                            color=TEXT_MUTED,
                            text_align=ft.TextAlign.CENTER,
                        ),
                    ],
                ),
                ft.ProgressRing(width=28, height=28, stroke_width=3, color=PRIMARY_BLUE),
            ],
        ),
    )


__all__ = ["build_splash_screen"]
