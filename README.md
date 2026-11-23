# Chess Position Recognition

Recognizes the full position on a real chessboard from a single photo. The pipeline splits the board into its 64 squares, classifies each one as empty or as one of the 12 piece types with a fine-tuned ResNet50, and rebuilds the position as a FEN string.

**Live demo:** [Try it on Streamlit](https://chessboard-square-classifier-iardjdxpqbydrrvjbyhqoy.streamlit.app/)

![Square grid extracted from a board image](docs/assets/sample_debug_grid.png)

## Highlights

- **~98.8% accuracy** on held-out games
- **Transfer learning:** ResNet50 pretrained on ImageNet, fine-tuned on square crops (13 classes)
- **Leave-one-game-out cross-validation**, so the model is always tested on a game it never saw in training
- **Augmentation for real-world photos:** color jitter for lighting changes, small rotations for camera angle
- **Out-of-distribution handling:** squares the model can't place are marked as unknown and drawn with a red X
- **Web demo:** upload a photo, get the predicted board, per-square confidence and the FEN

## How it works

```
photo ──► board grid ──► 64 square crops ──► ResNet50 ──► 8×8 labels ──► FEN
```

1. **Square extraction** (`dataset_tools/`): splits the board image into an 8×8 grid of crops. An optional OpenCV step detects the board and corrects perspective.
2. **Classification** (`src/model.py`): each crop is resized to 224×224 and classified as one of 13 classes (12 pieces + empty).
3. **Reconstruction** (`src/predict.py`, `dataset_tools/fen_utils.py`): the 64 predictions are assembled into an 8×8 board and converted to FEN.

## My contribution

I built the model and training side of the project:

- `src/dataset.py`: PyTorch `Dataset` that serves labeled square crops from the dataset manifest, with optional in-memory caching
- `src/model.py`: the ResNet-based classifier (ResNet18/34/50 backbones with a replaced classification head)
- `src/train.py`: the training loop, including augmentation, validation, checkpointing of the best model, and a CLI for hyperparameters
- `src/predict.py`: inference on a full board image, confidence thresholding, FEN conversion and SVG board rendering
- `test_extraction.py`: tests for square extraction and FEN round-trips on the real game data

## Training setup

| | |
|---|---|
| Backbone | ResNet50, ImageNet weights |
| Classes | 13 (6 white pieces, 6 black pieces, empty) |
| Input | 224×224 square crop, ImageNet normalization |
| Augmentation | ColorJitter (brightness 0.2, contrast 0.2, saturation 0.1), RandomRotation ±5° |
| Optimizer | Adam, lr = 0.001 |
| Batch size | 128 |
| Validation | Leave-one-game-out cross-validation |
| Hardware | Google Colab GPU ([notebook](notebooks/training_colab.ipynb)) |

## Run it locally

```bash
git clone https://github.com/Nikolkofman/chess-position-recognition.git
cd chess-position-recognition
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the trained weights from [Google Drive](https://drive.google.com/drive/folders/1NIhXsA4fIA4Ge7ooqqBfdrTkuDlXvCq9) and save them as `checkpoints/best_model.pth`.

```bash
# Predict a single image (add --save-viz to save an image with unknown squares marked)
python evaluate.py --image path/to/board.jpg --show-tensor

# Run the web demo at http://localhost:8501
streamlit run streamlit_app.py
```

From Python:

```python
import numpy as np
from PIL import Image
from evaluate import predict_board

board = predict_board(np.array(Image.open("board.jpg").convert("RGB")))
print(board)  # 8×8 tensor: 0–11 pieces, 12 empty, 13 unknown
```

### Training from scratch

```bash
python -m dataset_tools.make_dataset --data_root Data --out_root dataset_out
python src/train.py --manifest dataset_out/dataset_manifest.csv --model resnet50 --epochs 8 --batch_size 128
```

## Repository structure

```
├── src/                 # model, dataset, training and inference
├── dataset_tools/       # square extraction, board warp, FEN utilities, dataset builder
├── notebooks/           # Colab training notebook
├── checkpoints/         # put model weights here (not stored in git)
├── docs/assets/         # images for this README
├── evaluate.py          # predict_board(image) -> 8×8 tensor
├── demo.py              # command-line demo
├── streamlit_app.py     # web demo
└── create_compliant_dataset.py
```

## Tech stack

Python · PyTorch · torchvision · OpenCV · python-chess · pandas · Streamlit · Google Colab

## Team

Built as the final project for *Intro to Deep Learning* (Fall 2025) by:

- **Nikol Kofman:** dataset, model, training and inference
- **Ariel Shvarts:** dataset pipeline, evaluation API and Streamlit demo ([original repo](https://github.com/arikshvarts/chessboard-square-classifier))
- **Yaakov Gerelter**

The full commit history from the original team repository is preserved here.
