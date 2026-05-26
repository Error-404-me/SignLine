"""Parser for ESP32 glove packets containing flex and BNO055 IMU data."""

from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a numeric value to a range."""

    return max(lower, min(upper, value))


def _normalize_signed(value: float, scale: float) -> float:
    """Normalize signed sensor values to roughly -1..1."""

    if scale == 0:
        return 0.0
    return _clamp(value / scale, -1.0, 1.0)


def _parse_float_tuple(value: str, expected: int) -> tuple[float, ...] | None:
    """Parse a comma-separated tuple with the expected number of floats."""

    try:
        parts = tuple(float(part.strip()) for part in value.split(","))
    except ValueError:
        return None
    return parts if len(parts) == expected else None


@dataclass(frozen=True, slots=True)
class GloveReading:
    """One normalized ESP32 glove sensor reading."""

    flex: tuple[float, ...]
    euler: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    accelerometer: tuple[float, float, float]
    gyroscope: tuple[float, float, float]
    linear_accel: tuple[float, float, float]

    def to_feature_vector(self) -> list[float]:
        """Return the 26-value glove model feature vector."""

        return [
            *self.flex,
            *self.euler,
            *self.quaternion,
            *self.accelerometer,
            *self.gyroscope,
            *self.linear_accel,
        ]


class GloveParser:
    """Parse letter and sensor packets produced by the ESP32 glove firmware."""

    @staticmethod
    def is_sensor_packet(raw: str) -> bool:
        """Return True when a packet starts with the sensor payload prefix."""

        return raw.strip().startswith("flex:")

    @staticmethod
    def is_letter_packet(raw: str) -> bool:
        """Return True when a packet contains a one- or two-letter label."""

        value = raw.strip()
        return 1 <= len(value) <= 2 and value.isalpha()

    def parse(self, raw: str) -> GloveReading | None:
        """Parse a normalized sensor packet into a GloveReading."""

        packet = raw.strip()
        if not self.is_sensor_packet(packet):
            return None
        sections: dict[str, str] = {}
        for item in packet.split("|"):
            if ":" not in item:
                return None
            key, value = item.split(":", 1)
            sections[key.strip().lower()] = value.strip()

        flex_raw = _parse_float_tuple(sections.get("flex", ""), 10)
        euler_raw = _parse_float_tuple(sections.get("euler", ""), 3)
        quat_raw = _parse_float_tuple(sections.get("quat", ""), 4)
        accel_raw = _parse_float_tuple(sections.get("accel", ""), 3)
        gyro_raw = _parse_float_tuple(sections.get("gyro", ""), 3)
        laccel_raw = _parse_float_tuple(sections.get("laccel", ""), 3)
        if not all((flex_raw, euler_raw, quat_raw, accel_raw, gyro_raw, laccel_raw)):
            return None

        flex = tuple(_clamp(value / 4095.0, 0.0, 1.0) for value in flex_raw)
        euler = tuple(_normalize_signed(value, 360.0) for value in euler_raw)
        quaternion = tuple(_clamp(value, -1.0, 1.0) for value in quat_raw)
        accelerometer = tuple(_normalize_signed(value, 20.0) for value in accel_raw)
        gyroscope = tuple(_normalize_signed(value, 2000.0) for value in gyro_raw)
        linear_accel = tuple(_normalize_signed(value, 20.0) for value in laccel_raw)

        return GloveReading(
            flex=flex,
            euler=euler,  # type: ignore[arg-type]
            quaternion=quaternion,  # type: ignore[arg-type]
            accelerometer=accelerometer,  # type: ignore[arg-type]
            gyroscope=gyroscope,  # type: ignore[arg-type]
            linear_accel=linear_accel,  # type: ignore[arg-type]
        )


__all__ = ["GloveParser", "GloveReading"]
