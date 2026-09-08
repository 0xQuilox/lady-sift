const enabledEl = document.getElementById("enabled");
const thresholdEl = document.getElementById("threshold");
const thresholdVal = document.getElementById("thresholdVal");
const statusEl = document.getElementById("status");
const metricsEl = document.getElementById("metrics");

chrome.storage.local.get({ enabled: true, threshold: 0.5 }, (v) => {
  enabledEl.checked = v.enabled;
  thresholdEl.value = String(v.threshold);
  thresholdVal.textContent = Number(v.threshold).toFixed(2);
});

enabledEl.addEventListener("change", () => {
  chrome.storage.local.set({ enabled: enabledEl.checked });
});

thresholdEl.addEventListener("input", () => {
  const t = parseFloat(thresholdEl.value);
  thresholdVal.textContent = t.toFixed(2);
  chrome.storage.local.set({ threshold: t });
});

// Ping offscreen for readiness
chrome.runtime.sendMessage({ type: "LADY_SIFT_PING" }, (res) => {
  if (chrome.runtime.lastError) {
    statusEl.textContent = "Offscreen not ready";
    statusEl.className = "status warn";
    return;
  }
  if (res?.ready) {
    statusEl.textContent = `Model ready (${res.backend}) - 0.956 accuracy`;
    statusEl.className = "status ok";
  } else {
    statusEl.textContent = "Model loading…";
    statusEl.className = "status warn";
  }
});

// Show bundled metrics.json if web accessible
fetch(chrome.runtime.getURL("training/output/metrics.json"))
  .then((r) => (r.ok ? r.json() : fetch(chrome.runtime.getURL("models/tfjs_model/model.json")).then(() => ({ note: "metrics not bundled - see training/output/metrics.json" }))))
  .then((j) => (metricsEl.textContent = JSON.stringify(j, null, 2)))
  .catch(() => (metricsEl.textContent = "metrics.json not bundled (see training/output/metrics.json)"));
