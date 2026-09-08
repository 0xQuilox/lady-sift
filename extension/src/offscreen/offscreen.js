// Offscreen -> delegates to sandboxed iframe (sandbox allowed unsafe-eval for TF.js)
const pending = new Map();
let sandboxReady = false;
const sandboxEl = document.getElementById("sandbox");

window.addEventListener("message", (event) => {
  const msg = event.data;
  if (msg.type === "LADY_SIFT_RESULT") {
    const cb = pending.get(msg.id);
    if (cb) { pending.delete(msg.id); cb(msg); }
  }
  if (msg.type === "LADY_SIFT_PONG") {
    sandboxReady = true;
    const cb = pending.get("__ping__");
    if (cb) { pending.delete("__ping__"); cb(msg); }
  }
});

function sendToSandbox(msg) {
  sandboxEl.contentWindow.postMessage(msg, "*");
}

function pingSandbox() {
  return new Promise((resolve) => {
    pending.set("__ping__", resolve);
    sendToSandbox({ type: "LADY_SIFT_PING" });
    setTimeout(() => resolve({ ready: sandboxReady }), 3000);
  });
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "LADY_SIFT_PING") {
    pingSandbox().then(sendResponse);
    return true;
  }
  if (msg.type === "LADY_SIFT_CLASSIFY") {
    (async () => {
      try {
        const res = await fetch(msg.url);
        if (!res.ok) throw new Error(`fetch ${res.status} ${res.statusText}`);
        const blob = await res.blob();
        const bitmap = await createImageBitmap(blob);
        const id = msg.id;
        pending.set(id, (res) => sendResponse(res));
        // Transfer bitmap to sandbox
        try {
          sandboxEl.contentWindow.postMessage({ type: "LADY_SIFT_CLASSIFY_BITMAP", id, bitmap, threshold: msg.threshold, url: msg.url }, "*", [bitmap]);
        } catch {
          // Fallback if transfer fails (e.g. bitmap already closed)
          sandboxEl.contentWindow.postMessage({ type: "LADY_SIFT_CLASSIFY_BITMAP", id, bitmap, threshold: msg.threshold, url: msg.url }, "*");
        }
      } catch (e) {
        console.warn("[LadySift][offscreen] fetch failed", msg.url, e);
        sendResponse({ id: msg.id, url: msg.url, error: String(e) + (e.stack ? " " + e.stack.slice(0,500) : "") });
      }
    })();
    return true;
  }
});

// Warmup
sandboxEl.addEventListener("load", () => pingSandbox().catch(console.error));
