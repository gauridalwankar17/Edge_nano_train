"""MQTT helpers wrapping paho-mqtt for consistent usage across scripts."""

import json
import socket
from typing import Any, Callable, Optional

import paho.mqtt.client as mqtt


def create_client(
    client_id: str,
    host: str = "localhost",
    port: int = 1883,
    username: Optional[str] = None,
    password: Optional[str] = None,
    keepalive: int = 60,
    on_connect: Optional[Callable[[mqtt.Client, Any, dict, int], None]] = None,
    on_message: Optional[Callable[[mqtt.Client, Any, mqtt.MQTTMessage], None]] = None,
) -> mqtt.Client:
    """Creates and connects an MQTT client with optional callbacks."""
    client = mqtt.Client(client_id=client_id, clean_session=True)
    # Optional auth
    if username is not None and password is not None:
        client.username_pw_set(username, password)

    if on_connect is not None:
        client.on_connect = on_connect
    if on_message is not None:
        client.on_message = on_message

    # LWT to indicate this client went away unexpectedly
    client.will_set(
        topic=f"{client_id}/status",
        payload=json.dumps({"status": "offline"}),
        qos=0,
        retain=False,
    )

    # Connect
    try:
        client.connect(host, port, keepalive)
    except (ConnectionRefusedError, socket.gaierror) as exc:
        raise SystemExit(
            f"Failed to connect to MQTT broker at {host}:{port}. Ensure a broker is running. Error: {exc}"
        )

    # Start network loop in background
    client.loop_start()
    # Mark online
    client.publish(f"{client_id}/status", json.dumps({"status": "online"}), qos=0)
    return client


def publish_json(client: mqtt.Client, topic: str, payload: dict, qos: int = 0) -> None:
    """Safely publishes a JSON payload to a topic."""
    client.publish(topic, json.dumps(payload), qos=qos)

