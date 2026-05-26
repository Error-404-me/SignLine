"""Shared colors, spacing, and small theme helpers for the app."""

from __future__ import annotations

import flet as ft

PRIMARY_BLUE = "#2563EB"
PRIMARY_GREEN = "#16A34A"
BLUE_TINT = "#DBEAFE"
GREEN_TINT = "#DCFCE7"
BACKGROUND = "#F8FAFC"
SURFACE = "#FFFFFF"
SURFACE_MUTED = "#F1F5F9"
TEXT_PRIMARY = "#0F172A"
TEXT_SECONDARY = "#475569"
TEXT_MUTED = "#64748B"
BORDER = "#E2E8F0"
DANGER = "#DC2626"
WARNING = "#F59E0B"
SUCCESS = PRIMARY_GREEN

CARD_RADIUS = 8
PAGE_PADDING = 16
SECTION_GAP = 16
CENTER_ALIGNMENT = ft.Alignment(0, 0)

TAB_TITLES = {
    0: "Speech to Text",
    1: "Glove",
    2: "History",
    3: "Settings",
}


def tab_accent(index: int) -> str:
    """Return the active accent color for a navigation tab."""

    return PRIMARY_GREEN if index == 1 else PRIMARY_BLUE


def tab_tint(index: int) -> str:
    """Return a soft background tint for a navigation tab."""

    return GREEN_TINT if index == 1 else BLUE_TINT


def build_theme() -> ft.Theme:
    """Create the Material 3 theme used by the application."""

    return ft.Theme(
        color_scheme_seed=PRIMARY_BLUE,
        use_material3=True,
        visual_density=ft.VisualDensity.COMFORTABLE,
    )


def border_all(width: int | float, color: str) -> ft.Border:
    """Return a Flet border with the same side on all edges."""

    side = ft.BorderSide(width, color)
    return ft.Border(top=side, right=side, bottom=side, left=side)


__all__ = [
    "BACKGROUND",
    "BLUE_TINT",
    "BORDER",
    "CARD_RADIUS",
    "CENTER_ALIGNMENT",
    "DANGER",
    "GREEN_TINT",
    "PAGE_PADDING",
    "PRIMARY_BLUE",
    "PRIMARY_GREEN",
    "SECTION_GAP",
    "SUCCESS",
    "SURFACE",
    "SURFACE_MUTED",
    "TAB_TITLES",
    "TEXT_MUTED",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "WARNING",
    "border_all",
    "build_theme",
    "tab_accent",
    "tab_tint",
]
