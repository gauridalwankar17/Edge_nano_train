"""Edge inference MQTT subscriber.

Subscribes to an images topic, runs inference with the best trained model, and
publishes results to a predictions topic.

Usage:
  python edge_infer.py --broker localhost --port 1883 \
    --in-topic images/cifar10 --out-topic predictions/cifar10 \
    --checkpoint ./artifacts/best_model.pt --labels ./artifacts/labels.json

Input payload (from publisher):
{
  "image_b64": "...",
  "meta": {...}
}

Output payload:
{
  "prediction": {"index": int, "label": str, "confidence": float},
  "latency_ms": float,
  "meta": {...original meta...}
}
"""

import argparse
import json
from typing import Optional

import torch

from common.data import base64_to_pil, prepare_tensor_for_model
from common.inference import load_labels, load_model, logits_to_top1, predict_logits
from common.mqtt import create_client, publish_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge inference over MQTT")
    parser.add_argument("--broker", type=str, default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--in-topic", type=str, default="images/cifar10", help="Subscription topic")
    parser.add_argument("--out-topic", type=str, default="predictions/cifar10", help="Publish topic for predictions")
    parser.add_argument("--checkpoint", type=str, default="./artifacts/best_model.pt", help="Model checkpoint path")
    parser.add_argument("--labels", type=str, default="./artifacts/labels.json", help="Labels JSON path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    labels = load_labels(args.labels)
    model = load_model(args.checkpoint, device=device, num_classes=len(labels) if labels else 10)

    def on_connect(client, userdata, flags, rc):
        client.subscribe(args.in_topic)

    def on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8"))
            image_b64 = data.get("image_b64")
            meta = data.get("meta", {})
            if not image_b64:
                return
            image = base64_to_pil(image_b64)
            input_tensor = prepare_tensor_for_model(image)
            logits, latency_ms = predict_logits(model, input_tensor, device)
            pred = logits_to_top1(logits, labels)
            out = {"prediction": pred, "latency_ms": latency_ms, "meta": meta}
            publish_json(client, args.out_topic, out)
        except Exception as exc:
            # For robustness, avoid crashing on bad payloads
            publish_json(
                client,
                args.out_topic,
                {"error": str(exc), "meta": {"note": "bad payload"}},
            )

    client = create_client(
        client_id="edge_infer",
        host=args.broker,
        port=args.port,
        on_connect=on_connect,
        on_message=on_message,
    )

    try:
        # Keep running until interrupted
        import time

        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()


