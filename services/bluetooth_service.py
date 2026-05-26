"""Bluetooth and TCP glove I/O manager with Android-safe fallbacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import pickle
from queue import Empty, Queue
import socket
import threading
from typing import Any

from services.glove_parser import GloveParser, GloveReading

try:
    from jnius import autoclass

    BluetoothAdapter = autoclass("android.bluetooth.BluetoothAdapter")
    BluetoothDevice = autoclass("android.bluetooth.BluetoothDevice")
    BluetoothSocket = autoclass("android.bluetooth.BluetoothSocket")
    UUID = autoclass("java.util.UUID")
    ANDROID = True
except Exception:
    autoclass = None  # type: ignore[assignment]
    BluetoothAdapter = None  # type: ignore[assignment]
    BluetoothDevice = None  # type: ignore[assignment]
    BluetoothSocket = None  # type: ignore[assignment]
    UUID = None  # type: ignore[assignment]
    ANDROID = False

SPP_UUID = "00001101-0000-1000-8000-00805F9B34FB"
TCP_HOST = "127.0.0.1"
TCP_PORT = 9999


class BluetoothMode(Enum):
    """Incoming glove data mode."""

    LETTER = "letter"
    SENSOR = "sensor"


@dataclass(frozen=True, slots=True)
class BluetoothDeviceInfo:
    """Display-ready Bluetooth device metadata."""

    name: str
    address: str
    paired: bool = True


class BluetoothError(RuntimeError):
    """Raised when a Bluetooth or TCP operation fails."""


class BluetoothManager:
    """Manage Android RFCOMM, TCP simulator I/O, and glove packet callbacks."""

    def __init__(self, assets_dir: Path | None = None) -> None:
        """Create a Bluetooth manager and start the local TCP simulator server."""

        self.android = ANDROID
        self.assets_dir = assets_dir or Path("assets")
        self.mode = BluetoothMode.LETTER
        self.on_data: Callable[[str], None] | None = None
        self.on_reading: Callable[[GloveReading], None] | None = None
        self.connected_device: BluetoothDeviceInfo | None = None
        self._adapter: Any | None = None
        self._socket: Any | None = None
        self._input_stream: Any | None = None
        self._output_stream: Any | None = None
        self._send_queue: Queue[str] = Queue()
        self._sender_thread: threading.Thread | None = None
        self._receiver_thread: threading.Thread | None = None
        self._sender_stop = threading.Event()
        self._receiver_stop = threading.Event()
        self._parser = GloveParser()
        self._incoming_buffer = ""
        self._model: Any | None = None
        self._model_loaded = False
        self._lock = threading.RLock()

        self._tcp_server_socket: socket.socket | None = None
        self._tcp_client_socket: socket.socket | None = None
        self._tcp_server_thread: threading.Thread | None = None
        self._tcp_stop = threading.Event()
        self._tcp_lock = threading.RLock()

        if self.android:
            try:
                self._adapter = BluetoothAdapter.getDefaultAdapter()
            except Exception as exc:
                self.android = False
                raise BluetoothError(f"Unable to access Bluetooth adapter: {exc}") from exc
        self._start_tcp_server()

    @property
    def is_connected(self) -> bool:
        """Return whether an RFCOMM socket or TCP simulator is connected."""

        return self.connected_device is not None and (
            self._socket is not None or self._tcp_client_socket is not None
        )

    @property
    def status_text(self) -> str:
        """Return a user-facing Bluetooth/TCP connection status."""

        if self.connected_device and self._tcp_client_socket is not None:
            return f"Connected - {self.connected_device.name}"
        if self.connected_device and self._socket is not None:
            return f"Connected - {self.connected_device.name}"
        if self.android:
            return "Disconnected"
        return "Disconnected" #f"Waiting for PC simulator on localhost:{TCP_PORT}"

    def set_mode(self, mode: BluetoothMode) -> None:
        """Set whether incoming sensor packets are passed through or classified."""

        self.mode = mode

    def scan_devices(self) -> list[BluetoothDeviceInfo]:
        """Start discovery and return paired devices or the TCP simulator fallback."""

        devices: list[BluetoothDeviceInfo] = []
        if self.android and self._adapter is not None:
            try:
                if not self._adapter.isEnabled():
                    raise BluetoothError("Bluetooth is disabled. Turn it on in Android settings.")
                self._adapter.startDiscovery()
                paired = self._adapter.getBondedDevices()
                iterator = paired.iterator()
                while iterator.hasNext():
                    device = iterator.next()
                    name = device.getName() or "Unnamed device"
                    address = device.getAddress()
                    devices.append(BluetoothDeviceInfo(name=name, address=address, paired=True))
            except BluetoothError:
                raise
            except Exception as exc:
                raise BluetoothError(f"Bluetooth scan failed: {exc}") from exc

        devices.sort(key=lambda item: item.name.lower())
        if not devices:
            devices.append(
                BluetoothDeviceInfo(
                    name="PC Simulator",
                    address=f"tcp://localhost:{TCP_PORT}",
                    paired=False,
                )
            )
        return devices

    def connect(self, address: str, name: str | None = None) -> BluetoothDeviceInfo:
        """Connect to an Android Bluetooth device or expose the TCP fallback."""

        if address.startswith("tcp://"):
            info = BluetoothDeviceInfo(name=name or "PC Simulator", address=address, paired=False)
            with self._lock:
                if self._tcp_client_socket is not None:
                    self.connected_device = info
                return info

        if not self.android or self._adapter is None or UUID is None:
            raise BluetoothError("Bluetooth connection is supported only on Android.")
        with self._lock:
            self.disconnect()
            try:
                if not self._adapter.isEnabled():
                    raise BluetoothError("Bluetooth is disabled. Turn it on in Android settings.")
                device = self._adapter.getRemoteDevice(address)
                spp_uuid = UUID.fromString(SPP_UUID)
                socket_obj = device.createRfcommSocketToServiceRecord(spp_uuid)
                self._adapter.cancelDiscovery()
                socket_obj.connect()
                self._socket = socket_obj
                self._input_stream = socket_obj.getInputStream()
                self._output_stream = socket_obj.getOutputStream()
                info = BluetoothDeviceInfo(
                    name=name or device.getName() or "Bluetooth device",
                    address=address,
                    paired=True,
                )
                self.connected_device = info
                self._start_sender()
                self._start_receiver()
                return info
            except BluetoothError:
                self.disconnect()
                raise
            except Exception as exc:
                self.disconnect()
                raise BluetoothError(f"Could not connect to {address}: {exc}") from exc

    def send_text(self, text: str) -> None:
        """Queue recognised text for transmission over the connected output."""

        cleaned = text.strip()
        if not cleaned or not self.is_connected:
            return
        self._send_queue.put(cleaned)

    def disconnect(self) -> None:
        """Close active Bluetooth/TCP client sockets and stop sender/receiver loops."""

        with self._lock:
            self._sender_stop.set()
            self._receiver_stop.set()
            if (
                self._sender_thread
                and self._sender_thread.is_alive()
                and threading.current_thread() is not self._sender_thread
            ):
                self._sender_thread.join(timeout=0.8)
            if (
                self._receiver_thread
                and self._receiver_thread.is_alive()
                and threading.current_thread() is not self._receiver_thread
            ):
                self._receiver_thread.join(timeout=0.8)
            self._sender_thread = None
            self._receiver_thread = None
            self._sender_stop.clear()
            self._receiver_stop.clear()
            try:
                if self._output_stream is not None:
                    self._output_stream.close()
            except Exception:
                pass
            try:
                if self._input_stream is not None:
                    self._input_stream.close()
            except Exception:
                pass
            try:
                if self._socket is not None:
                    self._socket.close()
            except Exception:
                pass
            with self._tcp_lock:
                try:
                    if self._tcp_client_socket is not None:
                        self._tcp_client_socket.close()
                except Exception:
                    pass
                self._tcp_client_socket = None
            self._input_stream = None
            self._output_stream = None
            self._socket = None
            self.connected_device = None
            while not self._send_queue.empty():
                try:
                    self._send_queue.get_nowait()
                except Empty:
                    break

    def _start_sender(self) -> None:
        """Start the daemon thread that drains outgoing writes."""

        if self._sender_thread and self._sender_thread.is_alive():
            return
        self._sender_stop.clear()
        self._sender_thread = threading.Thread(
            target=self._sender_loop,
            name="bluetooth-sender",
            daemon=True,
        )
        self._sender_thread.start()

    def _start_receiver(self) -> None:
        """Start the Android Bluetooth receiver thread."""

        if self._receiver_thread and self._receiver_thread.is_alive():
            return
        self._receiver_stop.clear()
        self._receiver_thread = threading.Thread(
            target=self._receiver_loop,
            name="bluetooth-receiver",
            daemon=True,
        )
        self._receiver_thread.start()

    def _sender_loop(self) -> None:
        """Write queued messages to Bluetooth or TCP output streams."""

        while not self._sender_stop.is_set():
            try:
                text = self._send_queue.get(timeout=0.25)
            except Empty:
                continue
            payload = f"{text}\n".encode("utf-8")
            try:
                if self._output_stream is not None:
                    self._output_stream.write(payload)
                    self._output_stream.flush()
                with self._tcp_lock:
                    if self._tcp_client_socket is not None:
                        self._tcp_client_socket.sendall(payload)
            except Exception:
                self.disconnect()
                return

    def _receiver_loop(self) -> None:
        """Read incoming bytes from an Android Bluetooth input stream."""

        while not self._receiver_stop.is_set():
            try:
                if self._input_stream is None:
                    return
                value = self._input_stream.read()
                if value in (-1, None):
                    return
                self._feed_incoming_text(chr(int(value) & 0xFF))
            except Exception:
                self.disconnect()
                return

    def _start_tcp_server(self) -> None:
        """Start a localhost TCP server for PC simulator input."""

        if self._tcp_server_thread and self._tcp_server_thread.is_alive():
            return
        self._tcp_stop.clear()
        self._tcp_server_thread = threading.Thread(
            target=self._tcp_server_loop,
            name="glove-tcp-server",
            daemon=True,
        )
        self._tcp_server_thread.start()

    def _tcp_server_loop(self) -> None:
        """Accept localhost simulator clients and read glove packets."""

        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((TCP_HOST, TCP_PORT))
            server.listen(1)
            server.settimeout(0.5)
            self._tcp_server_socket = server
        except Exception:
            return

        while not self._tcp_stop.is_set():
            try:
                client, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with self._tcp_lock:
                if self._tcp_client_socket is not None:
                    try:
                        self._tcp_client_socket.close()
                    except Exception:
                        pass
                self._tcp_client_socket = client
                self.connected_device = BluetoothDeviceInfo(
                    name="PC Simulator",
                    address=f"tcp://localhost:{TCP_PORT}",
                    paired=False,
                )
                self._start_sender()
            threading.Thread(
                target=self._tcp_client_loop,
                args=(client,),
                name="glove-tcp-client",
                daemon=True,
            ).start()

    def _tcp_client_loop(self, client: socket.socket) -> None:
        """Read packets from one simulator TCP client."""

        try:
            client.settimeout(0.5)
            while not self._tcp_stop.is_set():
                try:
                    data = client.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    break
                self._feed_incoming_text(data.decode("utf-8", errors="ignore"))
        finally:
            with self._tcp_lock:
                if self._tcp_client_socket is client:
                    self._tcp_client_socket = None
                    if self.connected_device and self.connected_device.address.startswith("tcp://"):
                        self.connected_device = None
            try:
                client.close()
            except Exception:
                pass

    def _feed_incoming_text(self, text: str) -> None:
        """Frame incoming text into newline-delimited packets."""

        for char in text:
            if char in "\r\n":
                payload = self._incoming_buffer
                self._incoming_buffer = ""
                self._dispatch_payload(payload)
            else:
                self._incoming_buffer += char
                if self._incoming_buffer in {" ", "\b", "\x7f"}:
                    payload = self._incoming_buffer
                    self._incoming_buffer = ""
                    self._dispatch_payload(payload)

    def _dispatch_payload(self, payload: str) -> None:
        """Dispatch a letter, control, or full sensor packet to callbacks."""

        if payload in {" ", "\b", "\x7f"}:
            if self.on_data is not None:
                self.on_data(payload)
            return

        data = payload.strip()
        if not data:
            return
        if self._parser.is_letter_packet(data):
            if self.on_data is not None:
                self.on_data(data)
            return
        if self._parser.is_sensor_packet(data):
            reading = self._parser.parse(data)
            if reading is None:
                return
            if self.on_reading is not None:
                self.on_reading(reading)
            if self.mode == BluetoothMode.SENSOR and self.on_data is not None:
                predicted = self._predict_sensor_label(reading)
                if predicted:
                    self.on_data(predicted)

    def _predict_sensor_label(self, reading: GloveReading) -> str | None:
        """Predict a letter from a glove reading using assets/models/glove_rf.pkl."""

        model = self._load_glove_model()
        if model is None:
            return None
        vector = reading.to_feature_vector()
        target_size = int(getattr(model, "n_features_in_", len(vector)))
        if len(vector) < target_size:
            vector = [*vector, *([0.0] * (target_size - len(vector)))]
        elif len(vector) > target_size:
            vector = vector[:target_size]
        try:
            predicted = model.predict([vector])[0]
        except Exception:
            return None
        if isinstance(predicted, bytes):
            predicted = predicted.decode("utf-8", errors="ignore")
        return str(predicted).strip().upper() or None

    def _load_glove_model(self) -> Any | None:
        """Lazy-load and cache the glove RandomForest model."""

        if self._model_loaded:
            return self._model
        self._model_loaded = True
        model_path = self.assets_dir / "models" / "glove_rf.pkl"
        if not model_path.exists():
            return None
        try:
            with model_path.open("rb") as file:
                self._model = pickle.load(file)
        except Exception:
            self._model = None
        return self._model


__all__ = [
    "ANDROID",
    "SPP_UUID",
    "TCP_HOST",
    "TCP_PORT",
    "BluetoothAdapter",
    "BluetoothDevice",
    "BluetoothDeviceInfo",
    "BluetoothError",
    "BluetoothManager",
    "BluetoothMode",
    "BluetoothSocket",
]
