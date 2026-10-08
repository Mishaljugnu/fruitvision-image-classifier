# Architecture

```text
Browser
  │
  │ multipart image upload
  ▼
FastAPI
  │
  ├── validate image
  ├── resize + normalize
  ▼
MiniResNet
  │
  ├── class probabilities
  └── confidence threshold
  ▼
Prediction + confidence + latency
```

The model is loaded once and cached for subsequent inference requests.
