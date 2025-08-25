# Edge CNN + MQTT Demo

Train a compact CNN on CIFAR-10, publish images via MQTT, run edge inference,
and visualize live predictions with a Streamlit dashboard.

## Files
- `common/`: Shared utilities
  - `model.py`: CNN architecture
  - `data.py`: datasets, transforms, base64 encode/decode
  - `inference.py`: model loading and prediction
  - `mqtt.py`: MQTT client helpers
- `train_cnn.py`: Train on CIFAR-10, save `artifacts/best_model.pt` and labels
- `mqtt_publisher.py`: Publish CIFAR-10 test images as base64 JPEG
- `edge_infer.py`: Subscribe to images, run inference, publish predictions
- `dashboard.py`: Streamlit app to display live predictions
- `requirements.txt`: Python dependencies

## Prerequisites
- Python 3.9+
- An MQTT broker (e.g., Mosquitto). Example to run locally:

```bash
# Debian/Ubuntu
sudo apt-get update && sudo apt-get install -y mosquitto mosquitto-clients
sudo service mosquitto start
# Or run in a container
docker run -it --rm -p 1883:1883 eclipse-mosquitto
```

## Train
```bash
python train_cnn.py --data-dir ./datasets --epochs 15 --batch-size 128 --out-dir ./artifacts
```

## MQTT Flow
- Publisher topic: `images/cifar10`
- Subscriber publishes to: `predictions/cifar10`

Input payload (publisher):
```json
{"image_b64":"...","meta":{"source":"cifar10","index":0}}
```

Output payload (edge inference):
```json
{"prediction":{"index":0,"label":"airplane","confidence":0.99},"latency_ms":5.2,"meta":{}}
```

## Dashboard
```bash
streamlit run dashboard.py
```

## Quick-start
```bash
pip install -r requirements.txt
python train_cnn.py
python mqtt_publisher.py  # in one terminal
python edge_infer.py      # in another
streamlit run dashboard.py
```

## Notes
- TODO: Tune `--epochs`, `--lr`, and `--weight-decay` for higher accuracy.