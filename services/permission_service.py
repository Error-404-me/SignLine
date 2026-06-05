"""Cross-platform permission helpers — UI layer only."""

from __future__ import annotations

import asyncio

import flet as ft


async def ensure_bluetooth_permissions(
    page: ft.Page,
    permission_handler,
) -> bool:
    """Show explanation dialog then request Android Bluetooth permissions."""

    from services.stt_service import _is_android

    if not _is_android():
        return True

    try:
        from flet_permission_handler import Permission, PermissionStatus
    except ImportError:
        return True

    bluetooth_permissions = [
        Permission.BLUETOOTH_SCAN,
        Permission.BLUETOOTH_CONNECT,
        Permission.BLUETOOTH_ADVERTISE,
    ]

    statuses = {
        permission: await _check_permission(permission_handler, permission)
        for permission in bluetooth_permissions
    }
    if all(status == PermissionStatus.GRANTED for status in statuses.values()):
        return True

    result = {"granted": False}
    done = asyncio.Event()
    dialog: ft.AlertDialog

    async def on_grant(_) -> None:
        dialog.open = False
        page.update()
        any_permanently_denied = False
        for permission in bluetooth_permissions:
            if statuses[permission] == PermissionStatus.GRANTED:
                continue
            new_status = await permission_handler.request(permission)
            if new_status == PermissionStatus.PERMANENTLY_DENIED:
                any_permanently_denied = True
            elif new_status != PermissionStatus.GRANTED:
                result["granted"] = False
                done.set()
                return
        if any_permanently_denied:
            await permission_handler.open_app_settings()
            result["granted"] = False
        else:
            result["granted"] = True
        done.set()

    async def on_cancel(_) -> None:
        dialog.open = False
        page.update()
        result["granted"] = False
        done.set()

    dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text("Bluetooth Required"),
        content=ft.Text(
            "This app needs Bluetooth permission to scan for and connect to your "
            "ESP32 glove device. Please grant Bluetooth access when prompted."
        ),
        actions=[
            ft.TextButton("Cancel", on_click=on_cancel),
            ft.ElevatedButton("Grant Permission", on_click=on_grant),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )
    page.show_dialog(dialog)
    await done.wait()
    return result["granted"]


async def ensure_microphone_permission(
    page: ft.Page,
    permission_handler,
) -> bool:
    """Request microphone permission on Android. Always True on desktop."""

    from services.stt_service import _is_android

    if not _is_android():
        return True

    try:
        from flet_permission_handler import Permission, PermissionStatus
    except ImportError:
        return True

    status = await _check_permission(permission_handler, Permission.MICROPHONE)
    if status == PermissionStatus.GRANTED:
        return True

    new_status = await permission_handler.request(Permission.MICROPHONE)
    if new_status == PermissionStatus.PERMANENTLY_DENIED:
        await permission_handler.open_app_settings()
        return False
    return new_status == PermissionStatus.GRANTED


async def _check_permission(permission_handler, permission):
    """Check permission with the available flet_permission_handler API."""

    checker = getattr(permission_handler, "check", None)
    if checker is not None:
        return await checker(permission)
    return await permission_handler.get_status(permission)


__all__ = [
    "ensure_bluetooth_permissions",
    "ensure_microphone_permission",
]
