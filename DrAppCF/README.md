# Diabetic Retinopathy Stage Screening (Cloudflare Workers Edition)

A modern, fast web application for diabetic retinopathy screening built to run on the **Cloudflare Workers Free Tier** with zero server costs.

---

## ⚡ How It Works on Cloudflare Free Tier

Cloudflare Workers Free Tier has strict limits (128 MB RAM, 10ms CPU time, no native Python/PyTorch).

This project solves that elegantly with a **Dual-Engine Architecture**:
1. **Cloudflare Worker Edge (`src/index.js`)**: Serves the application globally via Cloudflare's edge CDN, provides `/api/model` and `/api/health` endpoints, and acts as an optional `/predict` proxy if an external inference server is connected.
2. **Client-Side In-Browser AI (`public/app.js`)**: Executes preprocessing (circular crop, contrast enhancement, unsharp mask), AI model evaluation, and Grad-CAM lesion heatmap rendering directly inside the user's browser using HTML5 Canvas & WebAssembly/WebGL.
3. **Hybrid Backend Support**: If you ever want to connect a dedicated GPU server, simply set `BACKEND_URL` in `wrangler.jsonc` and the worker will automatically proxy predictions!

---

## 📂 Project Structure

```
DrAppCF/
├── wrangler.jsonc       # Official Cloudflare Workers configuration
├── package.json         # Scripts for Wrangler CLI
├── src/
│   └── index.js         # Cloudflare Worker edge entry point
└── public/              # High-performance static assets
    ├── index.html       # Modern screening UI
    ├── styles.css       # Clean medical-grade design system
    ├── app.js           # Client logic, preprocessing & AI inference engine
    └── model.onnx       # (Optional) Exported ONNX model weights
```

---

## 🚀 Running Locally

1. Open a terminal in this directory:
   ```bash
   cd "h:\My Drive\DR_Project_Data\outputs\dr_app\DrAppCF"
   ```

2. Start the Cloudflare Worker local development server:
   ```bash
   npm run dev
   ```
   *or:*
   ```bash
   npx wrangler dev
   ```

3. Open `http://localhost:8787` in your browser.

---

## 🌐 Deploying to Cloudflare Workers (100% Free)

1. Make sure you are logged into Cloudflare Wrangler:
   ```bash
   npx wrangler login
   ```
   *(A browser window will open to authorize your free Cloudflare account)*

2. Deploy the Worker:
   ```bash
   npm run deploy
   ```
   *or:*
   ```bash
   npx wrangler deploy
   ```

3. Wrangler will output your live URL:
   `https://dr-app-cf.<your-subdomain>.workers.dev`

---

## 🧠 Optional: Exporting PyTorch Model to ONNX

To enable the browser to run your exact PyTorch trained weights (`best_dr_model.pth`):

1. Install ONNX dependencies:
   ```bash
   pip install onnx onnxscript
   ```

2. Run the export script in the parent folder:
   ```bash
   cd ..
   python export_onnx.py
   ```
   This will export `model.onnx` directly into `DrAppCF/public/model.onnx`.

3. Re-deploy with `npm run deploy`!
