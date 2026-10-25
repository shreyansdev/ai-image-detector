# AI Image Detector
A high-performance full-stack web application powered by a dual-engine PyTorch model system (**EfficientNet-B0** & **ForensicNet**) to identify and classify images as Real or AI-Generated with millisecond latency.

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch Version](https://img.shields.io/badge/PyTorch-2.5.1%2B-ee4c2c.svg)
![FastAPI Version](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)
![React Version](https://img.shields.io/badge/React-18.3%2B-20232a.svg)
![License MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## Overview
The rise of sophisticated deep generative models (such as Midjourney, DALL-E, and Stable Diffusion) has made it increasingly difficult to distinguish between real photographs and synthetic images. This project provides a full-stack, local-first machine learning application designed to inspect, analyze, and detect AI-generated images with high confidence and latency measured in single-digit milliseconds.

This application is built as an end-to-end ML serving system comprising:
* A client-side React 19 SPA with a reconciled stark ink-and-canvas design language (`DESIGN.md`, `theme.css`, `theme.js`), interactive 3D particle background (Three.js), single & batch drag-and-drop dropzones, and real-time model switching.
* A robust FastAPI REST backend providing single & batch inference endpoints, binary magic byte signature verification, sliding-window rate limiting, and CORS security.
* A dual-engine PyTorch model serving pipeline supporting **EfficientNet-B0** (highest accuracy) and **ForensicNet** (custom tri-stream spatial + frequency + noise residual architecture for ultra-fast GPU inference).

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend** | React 19, TypeScript, Vite 8, Reconciled Design System (`DESIGN.md`, `theme.css`, `theme.js`), Tailwind CSS v4, Three.js |
| **Backend** | FastAPI, Uvicorn, Pydantic, Python-dotenv |
| **Model & Inference** | PyTorch, torchvision, NumPy, PIL |
| **Training Hardware** | Intel Core i5-11400H, NVIDIA GeForce RTX 3050 (4GB VRAM), 16GB RAM |

---

## Model Details

* **EfficientNet-B0**: Fine-tuned transfer learning CNN (1280 → 512 → 2 head, Dropout 0.4). Backbone unfrozen at epoch 8 for deep fine-tuning. Optimized for maximum classification accuracy.
* **ForensicNet**: Custom tri-stream deep neural architecture combining spatial pixel analysis, noise residual extraction (SRM filters), and frequency domain (DCT) decomposition. Optimized for ultra-fast GPU throughput.
* **Dataset**: Combined dataset of Kaggle "AI Generated Images vs Real Images" and deepfake/face-swap samples (~238,000 images total) balanced across classes (`real` and `fake`).
* **Training Approach**: Mixed precision (AMP) training, automated worker benchmarking, and Adam optimizer with cosine/plateau learning rate schedules.

### Evaluation Results

| Metric | EfficientNet-B0 | ForensicNet |
| :--- | :--- | :--- |
| **Test Accuracy** | 97.82% | 94.83% |
| **AUC-ROC** | 0.9979 | 0.9903 |
| **Macro Precision** | 97.82% | 94.84% |
| **Macro Recall** | 97.82% | 94.83% |
| **Macro F1-Score** | 97.82% | 94.83% |
| **Parameter Count** | 4.66M | 1.05M |
| **Inference Speed (GPU)** | ~3.78 ms / image | ~1.78 ms / image |
| **Inference Speed (CPU)** | ~12.24 ms / image | ~5.60 ms / image |

### Visual Performance Diagnostics

#### Confusion Matrix & ROC Curve (EfficientNet-B0)
![Confusion Matrix](backend/ml/confusion_matrix.png)
![ROC Curve](backend/ml/roc_curve.png)

#### Confusion Matrix & ROC Curve (ForensicNet)
![ForensicNet Confusion Matrix](backend/ml/confusion_matrix_forensicnet.png)
![ForensicNet ROC Curve](backend/ml/roc_curve_forensicnet.png)

---

## Project Structure

```
ai-image-detector/
├── backend/
│   ├── app/
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   └── predict.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   └── inference.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── security.py
│   │   ├── main.py
│   │   └── __init__.py
│   ├── logs/
│   ├── ml/
│   │   ├── model_weights/
│   │   │   └── best_efficientnet_b0.pth
│   │   ├── clean_dataset.py
│   │   ├── dataset.py
│   │   ├── evaluate.py
│   │   ├── plot_training.py
│   │   ├── run_training_pipeline.py
│   │   └── train.py
│   ├── tests/
│   │   ├── conftest.py
│   │   └── test_api.py
│   ├── database.py
│   ├── main.py
│   ├── requirements.txt
│   ├── utils.py
│   └── verify_integration.py
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── components/
│   │   ├── services/
│   │   ├── App.tsx
│   │   ├── Background3D.tsx
│   │   ├── index.css
│   │   └── main.tsx
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── data/                    # Local dataset storage (gitignored)
└── README.md
```

---

## Getting Started

### Prerequisites
* Python 3.10+
* Node.js & npm
* NVIDIA GPU with CUDA drivers (Recommended but optional; fallback to CPU is automatic)

### Backend Setup

1. **Navigate to backend and create virtual environment**:
   ```bash
   cd backend
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install correct PyTorch GPU build (NVIDIA GPU users)**:
   If you have a CUDA-compatible GPU, reinstall the CUDA-enabled build of PyTorch for optimal performance:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Environment Configuration**:
   Create a `.env` file in the `backend/` directory copying `backend/.env` contents:
   ```env
   MODEL_WEIGHTS_PATH=ml/model_weights/best_efficientnet_b0.pth
   MAX_FILE_SIZE_MB=10
   ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
   ```

5. **Run FastAPI service**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   The backend API will run on `http://localhost:8000`.

### Frontend Setup

1. **Navigate to the frontend folder and install packages**:
   ```bash
   cd ../frontend
   npm install
   ```

2. **Configure Environment Variables**:
   Verify or create a `.env` file in the `frontend/` directory:
   ```env
   VITE_API_URL=http://localhost:8000
   ```

3. **Run local dev server**:
   ```bash
   npm run dev
   ```
   Open `http://localhost:5173` in your browser.

---

### Model Training & Pipeline (Optional)

To download the dataset and retrain the model locally:

1. Download the [Kaggle AI Generated vs Real Images Dataset](https://www.kaggle.com/datasets/salader/ai-generated-images-vs-real-images) and extract it to the `./data` directory.
2. Run the pipeline script to train the model, save metrics, generate diagnostic plots, and compile the final evaluator:
   ```bash
   python backend/ml/run_training_pipeline.py --data_path ./data --unfreeze_at 8
   ```

#### Training Flags:
* `--fresh_start`: Deletes the current training checkpoints and starts training fresh from epoch 1.
* `--benchmark_workers`: Benchmarks data loading speeds for worker values `[0, 2, 4, 6, 8]` on your CPU and configures the optimal count.
* `--amp`: Controls mixed precision training (enabled by default; set `--amp False` to disable).
* `--summary_only`: Skips model training and validation entirely and prints the final evaluation summary using the existing files.

---

## API Documentation

### Interactive Swagger Docs
When the backend is running, the interactive documentation is accessible at `http://localhost:8000/docs`.

### Endpoints

#### 1. Backend Health Check
* **Route**: `/health`
* **Method**: `GET`
* **Response**: `200 OK`
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "device": "cuda"
  }
  ```

#### 2. Classify Image
* **Route**: `/api/v1/predict`
* **Method**: `POST`
* **Request Content-Type**: `multipart/form-data`
* **Parameters**:
  * `image`: Binary image file (Accepted: `image/jpeg`, `image/png`, `image/webp`).
* **Success Response**: `200 OK`
  ```json
  {
    "label": "AI Generated",
    "confidence": 0.9854,
    "inference_time_ms": 3.71
  }
  ```
* **Error Responses**:
  * `413 Payload Too Large`: File exceeds the 10MB request size limit.
  * `415 Unsupported Media Type`: Invalid magic bytes signature or unsupported file format.
  * `422 Unprocessable Entity`: Corrupt image binary or decoding error.
  * `429 Too Many Requests`: Client IP exceeded the rate limit (30 requests/minute).
  * `503 Service Unavailable`: ML model failed to initialize or load during startup.

---

## Security Features

The backend implements several production-grade security measures:
* **Magic Byte Signature Check**: Validates the binary header signature of uploaded files to block extension-spoofed malicious uploads.
* **Payload Limitation**: Restricts maximum request size to 10MB at the middleware level to prevent denial of service (DoS) attacks.
* **Sliding Window Rate Limiter**: In-memory limiter restricting clients to 30 requests per minute per IP address.
* **CORS Restrictions**: Configurable CORS origins block cross-site execution from unauthorized web clients.
* **Secure HTTP Headers**: Automatically appends security-critical headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection`) to all responses.
* **Zero Secret Exposure**: No credentials, paths, or keys are hardcoded; all configuration is managed via `.env`.

---

## Testing

Run the pytest unit tests and integration verifier scripts inside the `backend` directory:
```bash
cd backend
# Run backend tests
pytest -v

# Run full integration checks
python verify_integration.py
```

---

## Limitations & Future Improvements
* **Local Session State**: The frontend uses standard session-based memory for past results; no database history is currently saved.
* **Generator Drift**: Classification accuracy is highest for the generators in the training set (Midjourney, DALL-E, Stable Diffusion) and may degrade for newer generator updates.
* **Batch Predictions**: Image uploads are currently single-file; future updates will support folder-level batch uploads.
* **Explainability (Grad-CAM)**: Future milestones include adding Grad-CAM heatmaps to visualize which image regions led the model to mark them as AI-generated.

---

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for more details.

---

## Acknowledgments
* Thanks to Kaggle and the dataset creator for compiling the balanced image dataset.
* Thanks to the developers of PyTorch, torchvision, FastAPI, and React for the core packages powering this application.
