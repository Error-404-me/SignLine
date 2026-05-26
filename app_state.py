"""Application state shared by tabs and services."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from services.bluetooth_service import BluetoothManager, BluetoothMode
from services.db_service import HistoryDatabase
from services.tts_service import TTSService


@dataclass(slots=True)
class AppSettings:
    """Mutable user settings for speech and device integrations."""

    speech_engine: str = "vosk"
    speech_model: str = "base"
    speech_language: str = "en"
    bluetooth_mode: str = BluetoothMode.LETTER.value
    auto_speak: bool = False
    mic_gain: float = 1.0
    denoise_enabled: bool = True
    highpass_enabled: bool = True
    highpass_cutoff_hz: float = 80.0
    denoise_prop_decrease: float = 0.8
    denoise_stationary: bool = True
    no_speech_threshold: float = 0.6
    vad_silence_ms: int = 500
    vad_threshold: float = 0.5
    available_speech_models: dict[str, str] = field(
        default_factory=lambda: {
            "tiny": "Whisper tiny",
            "base": "Whisper base",
        }
    )
    available_speech_engines: dict[str, str] = field(
        default_factory=lambda: {
            "vosk": "Vosk",
            "whisper": "Whisper",
        }
    )
    available_speech_languages: dict[str, str] = field(
        default_factory=lambda: {
            "en": "English",
            "tl": "Filipino",
        }
    )

    def whisper_model_name_or_path(self, assets_dir: Path) -> str:
        """Return a local CTranslate2 model folder or a faster-whisper model id."""

        local_model_dir = assets_dir / "models" / self.speech_model
        if local_model_dir.exists():
            return str(local_model_dir)
        return self.speech_model

    def vosk_model_path(self, assets_dir: Path) -> Path:
        """Return the local Vosk model folder for the selected speech language."""

        model_by_language = {
            "en": "vosk-model-small-en-us-0.15",
            "tl": "vosk-model-tl-ph-generic-0.6",
        }
        model_name = model_by_language.get(
            self.speech_language,
            model_by_language["en"],
        )
        return assets_dir / "models" / "vosk" / model_name

    def to_persisted_dict(self) -> dict[str, Any]:
        """Return user-selected settings suitable for local JSON storage."""

        return {
            "speech_engine": self.speech_engine,
            "speech_model": self.speech_model,
            "speech_language": self.speech_language,
            "bluetooth_mode": self.bluetooth_mode,
            "auto_speak": self.auto_speak,
            "mic_gain": self.mic_gain,
            "denoise_enabled": self.denoise_enabled,
            "highpass_enabled": self.highpass_enabled,
            "highpass_cutoff_hz": self.highpass_cutoff_hz,
            "denoise_prop_decrease": self.denoise_prop_decrease,
            "denoise_stationary": self.denoise_stationary,
            "no_speech_threshold": self.no_speech_threshold,
            "vad_silence_ms": self.vad_silence_ms,
            "vad_threshold": self.vad_threshold,
        }

    def apply_persisted_dict(self, values: dict[str, Any]) -> None:
        """Apply persisted user settings after validating known option keys."""

        speech_engine = str(values.get("speech_engine", self.speech_engine))
        if speech_engine in self.available_speech_engines:
            self.speech_engine = speech_engine

        speech_model = str(values.get("speech_model", self.speech_model))
        if speech_model in self.available_speech_models:
            self.speech_model = speech_model

        speech_language = str(values.get("speech_language", self.speech_language))
        if speech_language in self.available_speech_languages:
            self.speech_language = speech_language

        bluetooth_mode = str(values.get("bluetooth_mode", self.bluetooth_mode))
        if bluetooth_mode in {item.value for item in BluetoothMode}:
            self.bluetooth_mode = bluetooth_mode

        auto_speak = values.get("auto_speak", self.auto_speak)
        if isinstance(auto_speak, bool):
            self.auto_speak = auto_speak
        elif isinstance(auto_speak, str):
            self.auto_speak = auto_speak.lower() in {"1", "true", "yes", "on"}

        self.mic_gain = _float_setting(values, "mic_gain", self.mic_gain, 0.5, 3.0)
        self.denoise_enabled = _bool_setting(
            values,
            "denoise_enabled",
            self.denoise_enabled,
        )
        self.highpass_enabled = _bool_setting(
            values,
            "highpass_enabled",
            self.highpass_enabled,
        )
        self.highpass_cutoff_hz = _float_setting(
            values,
            "highpass_cutoff_hz",
            self.highpass_cutoff_hz,
            60.0,
            400.0,
        )
        self.denoise_prop_decrease = _float_setting(
            values,
            "denoise_prop_decrease",
            self.denoise_prop_decrease,
            0.1,
            1.0,
        )
        self.denoise_stationary = _bool_setting(
            values,
            "denoise_stationary",
            self.denoise_stationary,
        )
        self.no_speech_threshold = _float_setting(
            values,
            "no_speech_threshold",
            self.no_speech_threshold,
            0.3,
            0.95,
        )
        self.vad_silence_ms = _int_setting(
            values,
            "vad_silence_ms",
            self.vad_silence_ms,
            200,
            2000,
        )
        self.vad_threshold = _float_setting(
            values,
            "vad_threshold",
            self.vad_threshold,
            0.1,
            0.9,
        )


def _bool_setting(values: dict[str, Any], key: str, default: bool) -> bool:
    """Read a persisted boolean setting."""

    value = values.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return default


def _float_setting(
    values: dict[str, Any],
    key: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """Read and clamp a persisted float setting."""

    try:
        value = float(values.get(key, default))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _int_setting(
    values: dict[str, Any],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Read and clamp a persisted integer setting."""

    try:
        value = int(float(values.get(key, default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def load_app_settings(settings_path: Path) -> AppSettings:
    """Load user settings from JSON, falling back to defaults if unavailable."""

    settings = AppSettings()
    try:
        if settings_path.exists():
            with settings_path.open("r", encoding="utf-8") as file:
                values = json.load(file)
            if isinstance(values, dict):
                settings.apply_persisted_dict(values)
    except Exception:
        pass
    return settings


def save_app_settings(settings: AppSettings, settings_path: Path) -> None:
    """Persist user settings to a local JSON file."""

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with settings_path.open("w", encoding="utf-8") as file:
        json.dump(settings.to_persisted_dict(), file, indent=2)


@dataclass(slots=True)
class AppState:
    """Container for shared services and settings."""

    assets_dir: Path
    data_dir: Path
    settings: AppSettings
    db: HistoryDatabase
    bluetooth: BluetoothManager
    tts: TTSService = field(default_factory=TTSService)
    settings_path: Path | None = None

    def record_text(self, text: str, source: str) -> None:
        """Persist recognised text and send it over Bluetooth if connected."""

        cleaned = text.strip()
        if not cleaned:
            return
        db_source = "sign" if source == "glove" else source
        self.db.add_history(cleaned, db_source)
        self.bluetooth.send_text(cleaned)

    def save_settings(self) -> None:
        """Persist current user settings to local storage."""

        save_app_settings(
            self.settings,
            self.settings_path or (self.data_dir / "settings.json"),
        )


__all__ = [
    "AppSettings",
    "AppState",
    "load_app_settings",
    "save_app_settings",
]
