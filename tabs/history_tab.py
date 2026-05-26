"""History tab showing saved speech and sign recognitions."""

from __future__ import annotations

import flet as ft

from app_state import AppState
from components import copy_text, show_snack
from services.db_service import HistoryItem
from theme import (
    BORDER,
    CARD_RADIUS,
    CENTER_ALIGNMENT,
    DANGER,
    PAGE_PADDING,
    PRIMARY_BLUE,
    PRIMARY_GREEN,
    SECTION_GAP,
    SURFACE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    border_all,
)


class HistoryTab:
    """Flet view and controller for SQLite recognition history."""

    def __init__(self, page: ft.Page, state: AppState) -> None:
        """Create history controls."""

        self.page = page
        self.state = state
        self.search_bar = ft.SearchBar(
            bar_hint_text="Search recognised text",
            bar_leading=ft.Icon(ft.Icons.SEARCH, color=TEXT_MUTED),
            on_change=self._search_changed,
            controls=[],
        )
        self.list_view = ft.Column(
            expand=True,
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
        )
        self.empty_text = ft.Text(
            "No recognised text yet.",
            size=14,
            color=TEXT_MUTED,
            text_align=ft.TextAlign.CENTER,
        )
        self._view = self._build_view()

    def build(self) -> ft.Control:
        """Return the tab root control."""

        return self._view

    def on_visible(self) -> None:
        """Reload rows when the history tab becomes active."""

        self.refresh()

    def on_hidden(self) -> None:
        """Keep history data intact when leaving the tab."""

    def refresh(self) -> None:
        """Refresh the visible history rows."""

        items = self.state.db.list_history(search=self.search_bar.value)
        self.list_view.controls = (
            [self._build_tile(item) for item in items]
            if items
            else [
                ft.Container(
                    height=300,
                    alignment=CENTER_ALIGNMENT,
                    content=self.empty_text,
                )
            ]
        )
        self.empty_text.visible = not items
        self.page.update()

    def _build_view(self) -> ft.Control:
        """Create the responsive history tab layout."""

        content = ft.Column(
            expand=True,
            spacing=SECTION_GAP,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            controls=[
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    controls=[
                        ft.Text(
                            "Saved conversions",
                            size=16,
                            weight=ft.FontWeight.W_700,
                            color=TEXT_PRIMARY,
                        ),
                        ft.TextButton(
                            content=ft.Text("Clear all", color=DANGER, weight=ft.FontWeight.W_600),
                            icon=ft.Icons.DELETE_OUTLINE,
                            icon_color=DANGER,
                            on_click=self._confirm_clear,
                        ),
                    ],
                ),
                self.search_bar,
                ft.Container(
                    expand=True,
                    bgcolor=SURFACE,
                    border=border_all(1, BORDER),
                    border_radius=CARD_RADIUS,
                    padding=8,
                    content=self.list_view,
                ),
            ],
        )
        return ft.SafeArea(
            content=ft.Container(
                expand=True,
                padding=PAGE_PADDING,
                content=ft.ResponsiveRow(
                    expand=True,
                    alignment=ft.MainAxisAlignment.CENTER,
                    controls=[
                        ft.Container(
                            col={"xs": 12, "sm": 11, "md": 9, "lg": 7},
                            content=content,
                        )
                    ],
                ),
            )
        )

    def _build_tile(self, item: HistoryItem) -> ft.Control:
        """Build one history list tile."""

        is_speech = item.source == "speech"
        accent = PRIMARY_BLUE if is_speech else PRIMARY_GREEN
        icon = ft.Icons.MIC if is_speech else ft.Icons.BACK_HAND_OUTLINED
        title = item.text if len(item.text) <= 60 else f"{item.text[:57]}..."
        return ft.Container(
            bgcolor="#FFFFFF",
            border=border_all(1, BORDER),
            border_radius=8,
            content=ft.ListTile(
                leading=ft.Icon(icon, color=accent),
                title=ft.Text(title, color=TEXT_PRIMARY, weight=ft.FontWeight.W_600),
                subtitle=ft.Text(item.formatted_ts, color=TEXT_MUTED),
                trailing=ft.IconButton(
                    icon=ft.Icons.CONTENT_COPY,
                    icon_color=accent,
                    tooltip="Copy text",
                    on_click=lambda _, text=item.text: copy_text(self.page, text),
                ),
            ),
        )

    def _search_changed(self, _: ft.ControlEvent) -> None:
        """Filter history by the search field."""

        self.refresh()

    def _confirm_clear(self, _: ft.ControlEvent) -> None:
        """Show a confirmation dialog before clearing history."""

        def close_dialog() -> None:
            if self.page.pop_dialog() is None:
                dialog.open = False
            self.page.update()

        def clear_history(_: ft.ControlEvent) -> None:
            self.state.db.clear_history()
            close_dialog()
            self.refresh()
            show_snack(self.page, "History cleared.", bgcolor=PRIMARY_BLUE)

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Clear all history?"),
            content=ft.Text("This removes all saved speech and sign text from this device."),
            actions=[
                ft.TextButton("Cancel", on_click=lambda _: close_dialog()),
                ft.TextButton("Clear", on_click=clear_history),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.show_dialog(dialog)


__all__ = ["HistoryTab"]
