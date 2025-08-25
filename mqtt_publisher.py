"""MQTT image publisher.

Publishes CIFAR-10 test images (as base64 JPEG) to a topic for edge inference.

Usage:
  python mqtt_publisher.py --broker localhost --port 1883 --topic images/cifar10

The payload schema:
{
  "image_b64": "...",  # base64-encoded JPEG
  "meta": {"source": "cifar10", "index": int}
}
"""

import argparse
import time

from PIL import Image
import torchvision

from common.data import pil_to_jpeg_base64
from common.mqtt import create_client, publish_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish CIFAR-10 images over MQTT")
    parser.add_argument("--broker", type=str, default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--topic", type=str, default="images/cifar10", help="MQTT topic to publish images")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between publishes (seconds)")
    parser.add_argument("--limit", type=int, default=50, help="Number of images to publish (0 for all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    client = create_client(client_id="mqtt_publisher", host=args.broker, port=args.port)

    # CIFAR-10 test set without normalization for visualization/encoding
    test_ds = torchvision.datasets.CIFAR10(
        root="./datasets",
        train=False,
        download=True,
        transform=None,
    )

    count = 0
    for idx, (image_pil, label) in enumerate(test_ds):
        # torchvision returns PIL Image if transform=None
        if not isinstance(image_pil, Image.Image):
            # Convert tensor to PIL if needed
            image_pil = torchvision.transforms.ToPILImage()(image_pil)

        image_b64, w, h = pil_to_jpeg_base64(image_pil)
        payload = {
            "image_b64": image_b64,
            "meta": {"source": "cifar10", "index": idx, "width": w, "height": h},
        }
        publish_json(client, args.topic, payload)

        count += 1
        if args.limit > 0 and count >= args.limit:
            break
        time.sleep(max(0.0, args.delay))

    time.sleep(0.5)
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()


