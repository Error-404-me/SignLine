"""Reusable UI permission flows for platform features."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import flet as ft
from flet_permission_handler import Permission, PermissionHandler, PermissionStatus


BLUETOOTH_PERMISSIONS = (
    Permission.BLUETOOTH_SCAN,
    Permission.BLUETOOTH_CONNECT,
    Permission.BLUETOOTH_ADVERTISE,
)


async def ensure_bluetooth_permissions(
    page: ft.Page,
    permission_handler: PermissionHandler,
) -> bool:
    """Ensure Android Bluetooth permissions before UI starts Bluetooth work."""

    if not _is_android(page):
        return True

    missing = await _missing_bluetooth_permissions(permission_handler)
    if not missing:
        return True

    should_request = await _show_bluetooth_permission_dialog(page)
    if not should_request:
        return False

    for permission in missing:
        status = await permission_handler.get_status(permission)
        if status == PermissionStatus.GRANTED:
            continue
        if status == PermissionStatus.PERMANENTLY_DENIED:
            await permission_handler.open_app_settings()
            return False
        requested = await permission_handler.request(permission)
        if requested == PermissionStatus.PERMANENTLY_DENIED:
            await permission_handler.open_app_settings()
            return False

    missing_after_request = await _missing_bluetooth_permissions(permission_handler)
    if not missing_after_request:
        return True

    statuses = [
        await permission_handler.get_status(permission)
        for permission in missing_after_request
    ]
    if PermissionStatus.PERMANENTLY_DENIED in statuses:
        await permission_handler.open_app_settings()
    return False


async def _missing_bluetooth_permissions(
    permission_handler: PermissionHandler,
) -> list[Permission]:
    """Return Bluetooth permissions that are not granted."""

    missing: list[Permission] = []
    for permission in BLUETOOTH_PERMISSIONS:
        status = await permission_handler.get_status(permission)
        if status != PermissionStatus.GRANTED:
            missing.append(permission)
    return missing


async def _show_bluetooth_permission_dialog(page: ft.Page) -> bool:
    """Show the Bluetooth rationale dialog and return whether to proceed."""

    loop = asyncio.get_running_loop()
    decision: asyncio.Future[bool] = loop.create_future()
    dialog: ft.AlertDialog

    def close_with(value: bool) -> None:
        if not decision.done():
            decision.set_result(value)
        try:
            if page.pop_dialog() is None:
                dialog.open = False
        except Exception:
            dialog.open = False
        page.update()

    dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text("Bluetooth permission required"),
        content=ft.Text(
            "Bluetooth access is needed to scan for and connect to your ESP32 glove. "
            "The app uses it only for local device communication."
        ),
        actions=[
            ft.TextButton("Cancel", on_click=lambda _: close_with(False)),
            ft.TextButton("Grant Permission", on_click=lambda _: close_with(True)),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )
    page.show_dialog(dialog)
    page.update()
    return await decision


def _is_android(page: ft.Page) -> bool:
    """Return whether the current UI is running on Android."""

    platform: Any = getattr(page, "platform", None)
    return (
        sys.platform == "android"
        or platform == ft.PagePlatform.ANDROID
        or str(platform).lower().endswith("android")
    )


__all__ = ["BLUETOOTH_PERMISSIONS", "ensure_bluetooth_permissions"]
