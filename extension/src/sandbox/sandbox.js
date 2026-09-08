// Sandboxed TF.js - allowed unsafe-eval
let model = null;
const MODEL_URL = "../../models/tfjs_model/model.json";
const IMAGE_SIZE = 224;

async function ensureModel() {
  if (model) return model;
  await tf.ready();
  try { await tf.setBackend("webgl"); } catch {}
  await tf.ready();
  model = await tf.loadLayersModel(MODEL_URL);
  console.log("[LadySift][sandbox] model loaded", model.inputs[0].shape, "backend", tf.getBackend());
  return model;
}

function preprocessImageData(imageData) {
  return tf.tidy(() => {
    let t = tf.browser.fromPixels(imageData);
    t = tf.image.resizeBilinear(t, [IMAGE_SIZE, IMAGE_SIZE]);
    t = t.toFloat().div(127.5).sub(1);
    t = t.expandDims(0);
    return t;
  });
}

async function classifyImageBitmap(bitmap) {
  const m = await ensureModel();
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  const imageData = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  const input = preprocessImageData(imageData);
  const pred = m.predict(input);
  const prob = (await pred.data())[0];
  tf.dispose([input, pred]);
  return prob;
}

window.addEventListener("message", async (event) => {
  const msg = event.data;
  if (msg.type === "LADY_SIFT_CLASSIFY_BITMAP") {
    try {
      const bitmap = msg.bitmap;
      if (!bitmap) throw new Error("no bitmap");
      const fakeProb = await classifyImageBitmap(bitmap);
      bitmap.close?.();
      const label = fakeProb >= (msg.threshold ?? 0.5) ? "fake" : "real";
      window.parent.postMessage({ type: "LADY_SIFT_RESULT", id: msg.id, url: msg.url, fakeProb, label }, "*");
    } catch (e) {
      console.error("[LadySift][sandbox] classify error", e, e.stack);
      window.parent.postMessage({ type: "LADY_SIFT_RESULT", id: msg.id, url: msg.url, error: (e.message || String(e)) + " | stack: " + (e.stack || "").slice(0,1200) }, "*");
    }
  }
  if (msg.type === "LADY_SIFT_CLASSIFY") {
    // Fallback old path (direct fetch in sandbox) - keep for compatibility
    try {
      const res = await fetch(msg.url);
      if (!res.ok) throw new Error(`fetch ${res.status}`);
      const blob = await res.blob();
      const bitmap = await createImageBitmap(blob);
      const fakeProb = await classifyImageBitmap(bitmap);
      bitmap.close?.();
      const label = fakeProb >= (msg.threshold ?? 0.5) ? "fake" : "real";
      window.parent.postMessage({ type: "LADY_SIFT_RESULT", id: msg.id, url: msg.url, fakeProb, label }, "*");
    } catch (e) {
      window.parent.postMessage({ type: "LADY_SIFT_RESULT", id: msg.id, url: msg.url, error: String(e) + (e.stack ? " " + e.stack.slice(0,800) : "") }, "*");
    }
  }
  if (msg.type === "LADY_SIFT_PING") {
    try {
      await ensureModel();
      window.parent.postMessage({ type: "LADY_SIFT_PONG", ready: true, backend: tf.getBackend() }, "*");
    } catch (e) {
      window.parent.postMessage({ type: "LADY_SIFT_PONG", ready: false, error: String(e) }, "*");
    }
  }
});

ensureModel().catch(console.error);
