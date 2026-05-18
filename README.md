# Image Inpainting Inference Project

| Model | Colab version |
| ---   | ---      |
| VAE   | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aaTY2zrMmuE_2xG5CtrnQeb81fIYr7E0?usp=sharing) |
| Diffusion | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YSq4Q24aV8IyfI4yu13Z0HUIBuw6If-M?usp=sharing) |
| Flow matching | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1s-laGucP9LzkuSaLgLhYBzgCLUzwYxKZ?usp=sharing)| 

You can download the best pre-trained model weights from this [Google Drive folder](https://drive.google.com/drive/folders/1dq_FvdTiWQG-0ITLRGCJct3C3NlWRSIZ?usp=sharing) and place them in the `saved_models/` directory.

## Description

This project presents an image inpainting approach trained on 27,000 samples and validated on 100 samples from the [CelebA-HQ dataset (256×256)](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256). We resize to 64x64 and mask the central 21x21 region of each image (64 // 3) and train three generative models to reconstruct facial features such as eyes and nose.

![Data samples 1](/docs/image/data_samples_1.png)
![Data samples 2](/docs/image/data_samples_2.png)

### Model Architecture

All three models (VAE, Diffusion, and Flow Matching) are based on a UNet-style architecture with approximately **28M trainable parameters**.

The models operate on 7-channel input tensors combining:
- Ground truth image
- Masked condition image  
- Binary mask

During inference, the models use the masked image and mask to reconstruct the original content.

### Training Configuration

**VAE**
```
epochs: 100
beta: 5e-4
KL warmup epochs: 10
Optimizer: AdamW(lr=5e-5, weight_decay=1e-4)
Early stopping patience: 5
```
Model stopped at epoch 35 with the best validation loss of 0.06801

**Diffusion**
```
Noise Scheduler: DDPMScheduler
  - num_train_timesteps: 1000
  - beta_schedule: squaredcos_cap_v2
  - prediction_type: sample

Optimizer: AdamW(lr=5e-5, weight_decay=1e-4)
epochs: 100
MSE weight: 0.1
Early stopping patience: 5
```
Model stopped at epoch 31 with the best validation loss of 0.07912
*Note: We predict image samples instead of noise for better reconstruction quality.*

**Flow Matching**
```
epochs: 100
lambda_recon: 0.1
Optimizer: AdamW(lr=5e-5, weight_decay=1e-4)
Early stopping patience: 5
```
Model stopped at epoch 38 with the best validation loss of 0.09529

### Validation Results

**Ground Truth**
![Ground truth](/docs/image/ground_truth.png)

**VAE**
![VAE](/docs/image/vae.png)

**Diffusion (with 3 schedulers)**
- DDIM
![Diffusion DDIM](/docs/image/diffusion_ddim.png)
- DPM-Solver++
![Diffusion DPM](/docs/image/diffusion_dpm.png)
- UniPC
![Diffusion UniPC](/docs/image/diffusion_unipc.png)

**Flow Matching**
![Flow Matching](/docs/image/flow_matching.png)

### Observations
- **VAE**: Produces slightly blurred results with squinted eyes
- **Diffusion**: Tends to reconstruct complete faces with consistent eye color across all schedulers
- **Flow Matching**: Exhibits asymmetric eye reconstruction


## Inference

This repository is a production-style refactor of the original Colab notebook above, with AI support for code organization, refactoring, and building inference apps (FastAPI + Gradio).

The system works on 64x64 RGB images. For the default center-mask mode, the project masks the central 21x21 region of each image, which matches the training setup described above.

### 📁 Project Structure

```bash
.
├── src/                          # Source code
│   ├── api/                      # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py               # API endpoints
│   │   └── schemas.py            # Request/response models
│   ├── ml/                       # Model loading & inference
│   │   ├── __init__.py
│   │   ├── inference.py         # Inference pipeline and model cache
│   │   └── models.py            # Model architectures
│   ├── ui/                       # Gradio frontend
│   │   ├── __init__.py
│   │   └── app.py               # UI application
│   ├── utils/                    # Image helpers
│   │   ├── __init__.py
│   │   └── image_ops.py
│   └── __init__.py
├── notebooks/                    # Jupyter notebooks and exported scripts
├── saved_models/                 # Pre-trained model weights
│   ├── diffusion_best.pth
│   ├── flow_best.pth
│   └── vae_best.pth
├── requirements.txt              # Python dependencies
├── Makefile                      # Commands for easy execution
└── README.md                     # This file
```

### ⚙️ Installation

#### 1. Install Dependencies

```bash
make install
```

Or manually:

```bash
pip install -r requirements.txt
```

#### 2. Verify Model Files

Ensure all model weights exist in `saved_models/`:

```bash
ls -la saved_models/
```

### 🏃 Running the Application

#### Option 1: Run Both Servers (Recommended)

```bash
make run
```

This starts:
- API Server: http://127.0.0.1:8000
- Gradio UI: http://localhost:8501

#### Option 2: Run Separately

```bash
# Terminal 1: Start FastAPI
make run-api

# Terminal 2: Start Gradio UI
make run-ui
```

#### Option 3: Manual Run

```bash
# API Server
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

# Gradio UI
API_URL=http://127.0.0.1:8000 UI_HOST=localhost UI_PORT=8501 python -m src.ui.app
```

### 🌐 API Endpoints

- GET `/health`: health check
- POST `/inpaint`: run image inpainting inference

Request example:

```json
{
  "image_base64": "<base64_png>",
  "model_type": "vae",
  "mask_mode": "center",
  "x": 24,
  "y": 24,
  "w": 21,
  "h": 21,
  "scheduler_type": "ddim",
  "num_steps": 100,
  "device": "cpu"
}
```

### Notes

- `mask_mode="center"` uses the built-in central 21x21 mask.
- `mask_mode="custom"` uses the custom rectangle defined by `x`, `y`, `w`, and `h`.
- The UI preview and the API both use the same mask geometry helpers, so the displayed mask matches the backend inference input.
