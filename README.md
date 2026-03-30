# Hybrid DeepFake Detection System (CNN + Temporal Attention)

## Overview

PyTorch implementation of a lightweight deepfake detector combining spatial CNN features with temporal attention across sampled video frames.

Key features:
- ResNet18 / EfficientNet-B0 backbone
- Freeze backbone for fast transfer learning
- Per-frame CNN feature extraction + global attention aggregation
- Optional LSTM temporal modeling
- Mixed precision training support (GPU)
- Face detection (MTCNN fallback Haar Cascade)

## Project structure

- `src/model.py` -> model definition
- `src/dataset.py` -> dataset class and frame extraction / face crop
- `src/train.py` -> training pipeline
- `src/evaluate.py` -> evaluation pipeline
- `src/infer.py` -> single-video inference
- `requirements.txt` -> Python dependencies

## Dataset layout

Expected folder structure:

```
data_dir/
├── real/           # individual .jpg/.png images or .mp4 videos
│   ├── real_0.jpg
│   ├── real_1.jpg
│   └── ...
├── fake/           # individual .jpg/.png images or .mp4 videos
│   ├── fake_0.jpg
│   ├── fake_1.jpg
│   └── ...
```

Or for frame sequences:

```
data_dir/
├── real/
│   ├── video1/     # folder with frames
│   │   ├── frame_000.jpg
│   │   ├── frame_001.jpg
│   │   └── ...
│   └── video2.mp4  # or individual video files
├── fake/
│   └── ...
```

The dataset automatically detects:
- `.mp4/.avi` files → video mode (extracts frames)
- individual `.jpg/.png` images → image mode (repeats single image)
- folders with images → frame sequence mode

## Training

Install deps:

```bash
python -m pip install -r requirements.txt
```

Train:

```bash
python src/train.py --data-dir Dataset/Train --backbone resnet18 --frames 12 --batch-size 16 --epochs 12 --lr 1e-4 --mixed-precision --freeze-backbone --num-workers 0
```

**Note:** Use `--num-workers 0` to avoid DataLoader multiprocessing issues on Windows. Increase to 2-4 on Linux/Mac if you have enough RAM.

## Evaluation

```bash
python src/evaluate.py --data-dir Dataset/Validation --checkpoint checkpoint.pth --backbone resnet18 --frames 12
```

## Inference

```bash
python src/infer.py --input some_video.mp4 --checkpoint checkpoint.pth --backbone resnet18 --frames 12
```

## Web Interface (Streamlit)

Launch the interactive web app:

```bash
# Install additional dependencies
pip install streamlit Pillow

# Run the enhanced web app
streamlit run streamlit.py
```

**Features:**
- 🎨 **Modern UI** with custom styling and animations
- 📤 **Drag & drop** file upload (images/videos)
- ⚙️ **Advanced configuration** options in sidebar
- 📊 **Detailed analysis** with confidence visualization
- 🎯 **Real-time processing** with progress indicators
- 📱 **Responsive design** for desktop/mobile
- 🔍 **Face detection preview** (optional)
- 📋 **Comprehensive results** with explanations

**Supported formats:** JPG, PNG, MP4, AVI, MOV, WebM (Max 200MB)

### Deployment Options

**Local Development:**
```bash
streamlit run streamlit.py --server.port 8501 --server.address 0.0.0.0
```

**Production Deployment:**
- **Streamlit Cloud:** Upload to [share.streamlit.io](https://share.streamlit.io)
- **Heroku:** Use the provided `Procfile` and `requirements.txt`
- **Docker:** Build container with `Dockerfile`
- **AWS/Heroku:** Deploy as web service

**Note:** For production deployment, consider:
- Model optimization (ONNX/TensorRT)
- GPU instance for faster processing
- File size limits and timeout handling
- User authentication for sensitive deployments

## Programmatic Usage

Use the model in your own Python code:

```python
from app import load_model, preprocess_single_image, predict
from PIL import Image

# Load trained model
model = load_model('checkpoint.pth', backbone='resnet18')

# Process image
image = Image.open('image.jpg')
input_tensor = preprocess_single_image(image, num_frames=12)

# Get prediction
prediction, confidence, real_prob, fake_prob = predict(model, input_tensor)
print(f"Result: {prediction} ({confidence:.1%})")
```

Run the demo script:

```bash
python demo.py
```

This shows:
- Single image detection
- Batch processing examples
- Video processing (when video files are available)
