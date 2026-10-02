# Diabetic Retinopathy Detection & Clinical Screening System (`dr-detection-app`)

[![Cloudflare Workers](https://img.shields.io/badge/Deploy-Cloudflare_Workers-F38020?logo=cloudflare&logoColor=white)](https://workers.cloudflare.com/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-In--Browser%20AI-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning system designed for automated Diabetic Retinopathy (DR) grading and clinical decision support from retinal fundus photographs. 

This repository provides **two deployment architectures** tailored for both local/institutional medical workstations and zero-cost edge cloud deployments:
1. **Flask + PyTorch Web App**: Python-based clinical workstation server featuring deep PyTorch inference, circular cropping, CLAHE preprocessing, and Grad-CAM visual lesion heatmaps.
2. **Cloudflare Workers Web App (`DrAppCF`)**: Ultra-fast, zero-cloud-cost serverless edge deployment executing in-browser neural inference via WebAssembly/WebGL ONNX Runtime Web—ensuring **100% patient data privacy** (images never leave the patient's device).

---

## 🌟 Key Features

- **5-Stage ICDR Classification**:
  - `0 - No DR`: Normal healthy fundus
  - `1 - Mild DR`: Microaneurysms only
  - `2 - Moderate DR`: More than just microaneurysms, but less than severe
  - `3 - Severe DR`: Extensive intraretinal hemorrhages / venous beading
  - `4 - Proliferative DR`: Neovascularization / vitreous hemorrhage
- **Explainable AI (Grad-CAM)**: Generates localized attention heatmaps highlighting microaneurysms, hemorrhages, and exudates that influenced the model's prediction.
- **Retinal Image Preprocessing**: Automatic circular mask crop, square padding, CLAHE contrast enhancement, and unsharp masking.
- **Edge Deployment Ready**: In-browser client inference powered by ONNX Runtime Web requiring $0 compute infrastructure.

---

## 📁 Repository Structure

```text
dr-detection-app/
├── app.py                   # Flask web application server
├── dr_utils.py              # PyTorch inference, fundus preprocessing & Grad-CAM utilities
├── export_onnx.py           # PyTorch-to-ONNX model converter script
├── requirements.txt         # Python dependencies
├── best_dr_model.pth        # Trained EfficientNet-B3 DR model weights
├── templates/
│   └── index.html           # Flask clinical screening web UI
├── DrAppCF/                 # Cloudflare Workers Serverless & Edge App
│   ├── package.json         # Node.js / Wrangler scripts
│   ├── wrangler.jsonc       # Cloudflare Workers static asset & edge config
│   ├── src/
│   │   └── index.js         # Cloudflare Worker edge API router
│   └── public/
│       ├── index.html       # Modern responsive glassmorphism UI
│       ├── styles.css       # Medical-grade CSS design system
│       └── app.js           # Client-side preprocessing, ONNX engine & Grad-CAM
├── .gitignore               # Git ignore configuration
└── README.md                # Project documentation
```

---

## 🚀 Quick Start: Option 1 — Flask + PyTorch App

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (optional, runs on CPU as well)

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/JalithChamikara/dr-detection-app.git
cd dr-detection-app
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

> **Note**: By default, `app.py` loads `best_dr_model.pth` located in the root directory. To specify a custom model checkpoint, set the environment variable:
> ```bash
> export DR_MODEL="path/to/custom_model.pth"   # Linux/macOS
> $env:DR_MODEL="path/to/custom_model.pth"     # PowerShell
> ```

---

## ⚡ Quick Start: Option 2 — Cloudflare Workers App (`DrAppCF`)

The `DrAppCF` directory hosts the zero-cost edge version capable of running locally or deploying globally to Cloudflare Workers Free Tier.

### 1. Local Development
```bash
cd DrAppCF
npm install
npm run dev
```
Open **[http://localhost:8787](http://localhost:8787)** to test locally via Wrangler.

### 2. Deploy to Cloudflare Workers (Free Tier)
```bash
npx wrangler login
npm run deploy
```
Your app will be live globally at `https://dr-app-cf.<your-subdomain>.workers.dev`.

### 3. (Optional) Export PyTorch Model to ONNX
To export `best_dr_model.pth` to ONNX format for in-browser execution:
```bash
pip install onnx onnxscript
python export_onnx.py
```
The resulting `model.onnx` is placed in `DrAppCF/public/` for immediate browser inference.

---

## 🔬 Model & Preprocessing Pipeline

- **Architecture**: EfficientNet-B3 deep convolutional network fine-tuned for retinal grading.
- **Input Resolution**: 300 × 300 pixels (standardized RGB).
- **Clinical Preprocessing**:
  1. Automatic background removal & circular retinal disc detection
  2. Aspect ratio preservation with square zero-padding
  3. Green-channel CLAHE (Contrast Limited Adaptive Histogram Equalization)
  4. Image sharpening & normalization to ImageNet statistics

---

## ⚠️ Disclaimer

This software is developed for research, education, and screening prototype demonstration purposes only. It is not an FDA-approved medical diagnostic device and must not replace professional clinical judgment by a certified ophthalmologist or retinal specialist.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
