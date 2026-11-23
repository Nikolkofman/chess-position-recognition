# Model weights

Weights are too large for git. Download them from [Google Drive](https://drive.google.com/drive/folders/1NIhXsA4fIA4Ge7ooqqBfdrTkuDlXvCq9) and save the file here as:

```
checkpoints/best_model.pth
```

`evaluate.py` looks for a model in this order:

1. `checkpoints/best_model.pth`
2. `checkpoints/fold_1/best_model.pth`
3. `dataset_out/best_model_fold_1.pth`

`streamlit_app.py` downloads the weights automatically if none are found.

A checkpoint is a dict with `model_state_dict` (required), plus `optimizer_state_dict`, `epoch`, `val_acc` and `fold`.
