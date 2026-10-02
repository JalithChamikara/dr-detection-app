/**
 * Cloudflare Worker Backend for Diabetic Retinopathy Screening
 */

const CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'];

const NOTES = {
  0: 'No signs of diabetic retinopathy detected. Routine screening should continue as advised.',
  1: 'Early changes such as microaneurysms may be present. Follow-up screening is usually recommended.',
  2: 'More widespread changes are likely. An eye-care specialist should review this eye.',
  3: 'Extensive changes are likely. Prompt specialist review is important.',
  4: 'Signs of advanced disease are likely. Urgent specialist review is important.',
};

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // 1. Health check endpoint
    if (url.pathname === '/api/health') {
      return Response.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        environment: 'Cloudflare Workers (Edge)',
        has_backend_proxy: Boolean(env.BACKEND_URL),
      });
    }

    // 2. Model information endpoint
    if (url.pathname === '/api/model') {
      return Response.json({
        arch: env.MODEL_ARCH || 'efficientnet_b3',
        size: Number(env.IMAGE_SIZE) || 300,
        classes: Number(env.NUM_CLASSES) || 5,
        class_names: CLASS_NAMES,
        runtime: 'Cloudflare Worker + Client-side ONNX Web / Hybrid Proxy',
        notes: NOTES,
      });
    }

    // 3. Prediction endpoint
    if (url.pathname === '/predict' || url.pathname === '/api/predict') {
      if (request.method !== 'POST') {
        return Response.json({ error: 'Method not allowed. Use POST.' }, { status: 405 });
      }

      // If an external inference server is configured (e.g., Hugging Face, Render, Cloud Run, or local tunnel)
      if (env.BACKEND_URL) {
        try {
          const targetUrl = new URL('/predict', env.BACKEND_URL);
          const proxyReq = new Request(targetUrl, {
            method: 'POST',
            headers: request.headers,
            body: request.body,
          });
          const response = await fetch(proxyReq);
          return response;
        } catch (err) {
          return Response.json(
            { error: `Failed to forward prediction to backend (${err.message})` },
            { status: 502 }
          );
        }
      }

      // If no backend URL is bound, client-side browser inference is recommended
      return Response.json({
        status: 'client_side_ready',
        message: 'Cloudflare Workers free tier runs on V8 edge isolates. Real-time inference is handled directly in your browser using ONNX Web, or you can set a BACKEND_URL variable in wrangler.jsonc to proxy to a dedicated GPU/CPU server.',
        class_names: CLASS_NAMES,
        notes: NOTES,
      });
    }

    // 4. Default: Cloudflare Workers Static Assets
    if (env.ASSETS) {
      return env.ASSETS.fetch(request);
    }

    return new Response('Not Found', { status: 404 });
  },
};
