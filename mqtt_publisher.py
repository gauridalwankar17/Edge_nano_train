import argparse
import csv
import json
import time
from pathlib import Path

import paho.mqtt.client as mqtt


def publish_rows(csv_path: str, topic: str = "pump/vibration", host: str = "localhost", port: int = 1883, rate_hz: float = 50.0):
    client = mqtt.Client()
    client.connect(host, port, 60)
    client.loop_start()

    path = Path(csv_path)
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        period = 1.0 / rate_hz if rate_hz > 0 else 0.0
        for row in reader:
            payload = json.dumps(row)
            client.publish(topic, payload)
            if period > 0:
                time.sleep(period)

    client.loop_stop()
    client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Publish sensor CSV rows to MQTT topic pump/vibration")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file to stream")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", type=str, default="pump/vibration")
    parser.add_argument("--rate", type=float, default=50.0, help="Rows per second")
    args = parser.parse_args()

    publish_rows(args.csv, topic=args.topic, host=args.host, port=args.port, rate_hz=args.rate)


if __name__ == "__main__":
    main()


