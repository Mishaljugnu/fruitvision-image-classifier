# Model files

The trained FruitVision checkpoint is intentionally not included in this repository.

Expected files:

- `resnet_manual.py` — the `MiniResNet` architecture used to train the checkpoint.
- `resnet_fruit.pth` — the trained model weights.

Place both files in this directory before running the application locally.

The `.pth` checkpoint is ignored by Git because trained binary weights can be large. If you publish the weights separately, update `app/model_loader.py` to load them from that location.
