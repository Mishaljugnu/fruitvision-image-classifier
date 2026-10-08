# FruitVision

**Image classification with a custom MiniResNet model and FastAPI inference API.**

FruitVision classifies uploaded fruit images into three trained categories — **apple, durian, and banana** — and returns the prediction, confidence, and inference latency.

## What I built

- Custom CNN-based MiniResNet classifier
- Image preprocessing with resizing and ImageNet normalization
- Confidence-based rejection for low-confidence predictions
- FastAPI inference endpoint
- Browser-based upload and prediction interface
- Model evaluation experiments comparing AlexNet and MiniResNet variants

## How it works

```text
Image upload
    ↓
Resize + normalize
    ↓
MiniResNet
    ↓
Softmax probabilities
    ↓
60% confidence threshold
    ↓
Prediction / Unknown
```

`unknown` means the highest model probability is below the configured threshold. It is a confidence-based rejection rule, not a separately trained "other" class.

## Experiments

The `experiments/` directory contains the original evaluation visualizations:

- `alexnet.png`
- `miniresnet.png`
- `miniresnet2.png`
- `improvement.png`
- `training.png`

## Local setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The application expects the trained model files below:

```text
models/
├── resnet_manual.py
└── resnet_fruit.pth
```

They are intentionally not included in this archive yet because the source ZIP did not contain them. See `models/README.md`.

Run:

```bash
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000`.

## API

### `POST /predict`

Upload an image using the `file` multipart field.

```json
{
  "prediction": "banana",
  "confidence": 94.2,
  "latency_ms": 18.7,
  "threshold": 0.6
}
```

### `GET /health`

Reports whether the trained checkpoint is available.

## Deployment

`api/index.py` exposes the FastAPI application for Vercel's Python runtime. Before deploying, add the missing model architecture and trained checkpoint, then verify the PyTorch package/model size and cold-start performance in the target runtime.

For heavier ML inference, a dedicated Python inference host may be a better production choice than serverless deployment.

## Limitations

- Only three fruit classes are supported.
- The 60% threshold is a confidence rule, not formal open-set recognition.
- Accuracy depends on the training data and image conditions.
- The trained checkpoint is not included in this source archive yet.

## License

MIT License. See `LICENSE`.
