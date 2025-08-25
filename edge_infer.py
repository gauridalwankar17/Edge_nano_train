import argparse
import json
from collections import deque
from typing import Deque, Dict, List

import numpy as np
import paho.mqtt.client as mqtt
import torch


CLASS_NAMES = ["NORMAL", "WARNING", "FAILURE"]


def load_model(model_path: str) -> torch.jit.ScriptModule:
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Edge inference subscriber for pump vibration")
    parser.add_argument("--model", type=str, default="anomaly_cnn.pt")
    parser.add_argument("--subscribe", type=str, default="pump/vibration")
    parser.add_argument("--publish", type=str, default="pump/alert")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--feature-keys", type=str, nargs="*", default=None, help="Explicit feature keys to read from messages")
    args = parser.parse_args()

    model = load_model(args.model)

    # Infer expected number of features from first layer
    if hasattr(model, "num_features"):
        num_features = int(model.num_features)  # type: ignore[attr-defined]
    else:
        # fallback: attempt to read from first conv weight shape
        try:
            first_w = next(p for name, p in model.named_parameters() if "features.0.weight" in name)
            num_features = int(first_w.shape[1])
        except Exception:
            num_features = 1

    buffer: Deque[List[float]] = deque(maxlen=args.window_size)

    pub_client = mqtt.Client()
    pub_client.connect(args.host, args.port, 60)
    pub_client.loop_start()

    def on_message(client, userdata, msg):
        try:
            data: Dict[str, str] = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return

        # Select features
        if args.feature-keys:
            keys = args.feature-keys
        else:
            # Heuristic: take first num_features numeric fields
            numeric_items = []
            for k, v in data.items():
                try:
                    numeric_items.append((k, float(v)))
                except Exception:
                    continue
            numeric_items = numeric_items[:num_features]
            keys = [k for k, _ in numeric_items]

        row_vals: List[float] = []
        for k in keys:
            try:
                row_vals.append(float(data[k]))
            except Exception:
                row_vals.append(0.0)

        if len(row_vals) == 0:
            return

        buffer.append(row_vals)
        if len(buffer) < args.window_size:
            return

        X = np.array(buffer, dtype=np.float32)  # (T, F)
        # z-score per window
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-6
        Xn = (X - mean) / std
        # shape to (1, C=F, T)
        x_tensor = torch.from_numpy(Xn.T).unsqueeze(0)
        with torch.no_grad():
            logits = model(x_tensor)
            pred = int(torch.argmax(logits, dim=1).item())
            conf = float(torch.softmax(logits, dim=1)[0, pred].item())

        alert = {
            "prediction": CLASS_NAMES[pred],
            "confidence": round(conf, 4),
        }
        pub_client.publish(args.publish, json.dumps(alert))

    sub_client = mqtt.Client()
    sub_client.on_message = on_message
    sub_client.connect(args.host, args.port, 60)
    sub_client.subscribe(args.subscribe)
    sub_client.loop_forever()


if __name__ == "__main__":
    main()


