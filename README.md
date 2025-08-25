## edge_nano_twin

Lightweight edge ML pipeline for pump sensor anomaly classification with a tiny 1D CNN and MQTT.

### Setup

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Configure Kaggle API for dataset download:
   - Place `kaggle.json` in `~/.kaggle/kaggle.json` with your API credentials,
     or set env vars `KAGGLE_USERNAME` and `KAGGLE_KEY`.

### Download data

```bash
python -c "from edge_nano_twin.download_data import download_kaggle_dataset; download_kaggle_dataset()"
```

This creates a `data/` folder with the Kaggle dataset `nphantawee/pump-sensor-data`.

### Preprocess, train, and export model

The training script performs preprocessing (256-sample windows, z-score per window, class balancing), trains a tiny 1D CNN with early stopping, and exports a TorchScript model `anomaly_cnn.pt`.

```bash
python -m edge_nano_twin.train --data-dir data --output anomaly_cnn.pt --window-size 256
```

The exported TorchScript model is designed to be under 1.5 MB.

### MQTT real-time pipeline

Start an MQTT broker (e.g., Mosquitto) locally on port 1883.

- Publisher: stream CSV rows to `pump/vibration`

```bash
python mqtt_publisher.py --csv data/any_sensor_file.csv --topic pump/vibration --rate 50
```

- Edge inferencer: subscribe to `pump/vibration`, publish alerts to `pump/alert`

```bash
python edge_infer.py --model anomaly_cnn.pt --subscribe pump/vibration --publish pump/alert
```

Messages published to `pump/alert` have the schema:

```json
{"prediction": "NORMAL|WARNING|FAILURE", "confidence": 0.0-1.0}
```

### Tests

Run tests to validate model size (<1.5 MB) and latency (<20 ms p95 on CPU):

```bash
pytest -q
```

If `anomaly_cnn.pt` is not present, tests will generate a tiny fallback scripted model to complete the checks.

### Project tree

```
edge_nano_twin/
  __init__.py
  download_data.py
  preprocess.py
  model.py
  train.py
edge_infer.py
mqtt_publisher.py
tests/
  test_size_latency.py
requirements.txt
README.md
```

