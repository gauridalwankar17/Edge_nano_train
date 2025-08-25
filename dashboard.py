"""Streamlit dashboard for live predictions over MQTT.

Runs a small MQTT client in a thread to listen for predictions and displays
the latest results with confidence and simple stats.

Usage:
  streamlit run dashboard.py
"""

import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import streamlit as st

from common.mqtt import create_client


@dataclass
class PredictionEvent:
    timestamp: float
    label: str
    confidence: float
    meta: Dict[str, Any]


def mqtt_listener(broker: str, port: int, topic: str, out_queue: "queue.Queue[PredictionEvent]") -> None:
    def on_connect(client, userdata, flags, rc):
        client.subscribe(topic)

    def on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8"))
            if "prediction" in data:
                pred = data["prediction"]
                evt = PredictionEvent(
                    timestamp=time.time(),
                    label=str(pred.get("label", "?")),
                    confidence=float(pred.get("confidence", 0.0)),
                    meta=data.get("meta", {}),
                )
                out_queue.put(evt)
        except Exception:
            pass

    client = create_client(
        client_id="dashboard",
        host=broker,
        port=port,
        on_connect=on_connect,
        on_message=on_message,
    )

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


def main() -> None:
    st.set_page_config(page_title="Edge Predictions Dashboard", layout="centered")
    st.title("Edge Predictions Dashboard")

    broker = st.sidebar.text_input("MQTT Broker", value="localhost")
    port = st.sidebar.number_input("Port", min_value=1, value=1883, step=1)
    topic = st.sidebar.text_input("Predictions Topic", value="predictions/cifar10")

    status_placeholder = st.empty()
    latest_placeholder = st.empty()
    chart_placeholder = st.empty()

    # Thread-safe queue for incoming events
    event_queue: "queue.Queue[PredictionEvent]" = queue.Queue()

    # Start listener thread once per session
    if "listener_started" not in st.session_state:
        st.session_state.listener_started = True
        thread = threading.Thread(
            target=mqtt_listener, args=(broker, port, topic, event_queue), daemon=True
        )
        thread.start()

    # Keep recent confidences for chart
    confidences = st.session_state.get("confidences", [])

    # Poll the queue non-blocking
    try:
        while True:
            evt = event_queue.get_nowait()
            st.session_state["last_evt"] = evt
            confidences.append(evt.confidence)
            if len(confidences) > 100:
                confidences.pop(0)
    except queue.Empty:
        pass

    st.session_state["confidences"] = confidences

    # Status
    status_placeholder.info(f"Listening to `{topic}` on `{broker}:{port}`")

    # Latest event
    if "last_evt" in st.session_state:
        evt: PredictionEvent = st.session_state["last_evt"]
        latest_placeholder.markdown(
            f"**Latest:** {evt.label}  |  confidence={evt.confidence:.3f}  |  time={time.strftime('%X', time.localtime(evt.timestamp))}"
        )
    else:
        latest_placeholder.warning("Waiting for predictions...")

    # Simple line chart of confidence
    if confidences:
        chart_placeholder.line_chart(confidences)


if __name__ == "__main__":
    main()


