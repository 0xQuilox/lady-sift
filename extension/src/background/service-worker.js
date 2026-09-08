// Service worker - router between content/popup and offscreen TF.js
const OFFSCREEN_URL = "src/offscreen/offscreen.html";
let creatingOffscreen = null;

async function hasOffscreen() {
  if (!chrome.offscreen) return false;
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
  });
  return contexts.length > 0;
}

async function ensureOffscreen() {
  if (await hasOffscreen()) return;
  if (creatingOffscreen) await creatingOffscreen;
  else {
    creatingOffscreen = chrome.offscreen.createDocument({
      url: OFFSCREEN_URL,
      reasons: ["BLOBS"],
      justification: "Run TensorFlow.js locally for image classification",
    });
    await creatingOffscreen;
    creatingOffscreen = null;
  }
}

// Init offscreen on install
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({ enabled: true, threshold: 0.5 });
  ensureOffscreen().catch(console.error);
});

chrome.runtime.onStartup.addListener(() => ensureOffscreen().catch(console.error));

// Proxy classify requests to offscreen
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "LADY_SIFT_CLASSIFY") {
    ensureOffscreen()
      .then(() => chrome.runtime.sendMessage(msg))
      .then(sendResponse)
      .catch((e) => sendResponse({ id: msg.id, error: String(e) }));
    return true;
  }
  if (msg.type === "LADY_SIFT_PING") {
    ensureOffscreen()
      .then(() => chrome.runtime.sendMessage(msg))
      .then(sendResponse)
      .catch(() => sendResponse({ ready: false }));
    return true;
  }
});
