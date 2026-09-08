(() => {
  const PROCESSED_ATTR = "data-lady-sift";
  const BADGE_CLASS = "lady-sift-badge";
  const THRESHOLD_DEFAULT = 0.5;
  const MAX_PER_PAGE = 30;
  const MIN_SIZE = 80;

  let threshold = THRESHOLD_DEFAULT;
  let enabled = true;

  const style = document.createElement("style");
  style.textContent = `
    .${BADGE_CLASS} {
      position: absolute;
      top: 6px; left: 6px;
      padding: 3px 7px;
      border-radius: 10px;
      font: 12px/1.2 system-ui, sans-serif;
      color: #fff;
      pointer-events: none;
      z-index: 2147483647;
      box-shadow: 0 1px 6px rgba(0,0,0,.4);
    }
    .${BADGE_CLASS}[data-label="fake"] { background: #c0392b; }
    .${BADGE_CLASS}[data-label="real"] { background: #27ae60; }
    .lady-sift-wrapper { position: relative !important; display: inline-block; }
    img[${PROCESSED_ATTR}="clickable"] { cursor: pointer; outline: 2px dashed #3498db; outline-offset: 2px; }
    .lady-sift-hint {
      position: absolute; bottom: 6px; right: 6px;
      background: rgba(52,152,219,0.9); color: #fff;
      font: 10px/1 system-ui; padding: 2px 5px; border-radius: 8px;
      pointer-events: none; z-index: 2147483647;
    }
  `;
  document.documentElement.appendChild(style);

  chrome.storage.local.get({ enabled: true, threshold: 0.5 }, (v) => {
    enabled = v.enabled;
    threshold = v.threshold;
    if (enabled) markClickable();
  });
  chrome.storage.onChanged.addListener((changes) => {
    if (changes.threshold) threshold = changes.threshold.newValue;
    if (changes.enabled) {
      enabled = changes.enabled.newValue;
      if (enabled) markClickable();
      else clearBadges();
    }
  });

  function clearBadges() {
    document.querySelectorAll(`.${BADGE_CLASS}`).forEach((b) => b.remove());
    document.querySelectorAll(`.${BADGE_CLASS}-hint`).forEach((b) => b.remove());
    document.querySelectorAll(`[${PROCESSED_ATTR}]`).forEach((el) => {
      el.removeAttribute(PROCESSED_ATTR);
      el.style.cursor = "";
    });
  }

  function markClickable() {
    if (!enabled) return;
    for (const img of document.querySelectorAll("img")) {
      if (img.hasAttribute(PROCESSED_ATTR) || img.dataset.ladySiftBound) continue;
      const w = img.naturalWidth || img.width;
      const h = img.naturalHeight || img.height;
      if (w < MIN_SIZE || h < MIN_SIZE) continue;
      const url = img.currentSrc || img.src;
      if (!url || !/^https?:|^blob:|^data:image/.test(url)) continue;
      img.dataset.ladySiftBound = "1";
      img.setAttribute(PROCESSED_ATTR, "clickable");
      img.style.cursor = "pointer";
      img.title = "Click to check with Lady Sift";
      img.addEventListener("click", onClickImage);
    }
  }

  function onClickImage(e) {
    if (!enabled) return;
    // Allow Ctrl/Cmd+click to open image normally
    if (e.ctrlKey || e.metaKey) return;
    e.preventDefault();
    e.stopPropagation();
    const img = e.currentTarget;
    const url = img.currentSrc || img.src;
    if (!url) return;
    // Toggle off if already classified
    const wrapper = img.closest(".lady-sift-wrapper");
    if (wrapper && wrapper.querySelector(`.${BADGE_CLASS}`)) {
      wrapper.querySelectorAll(`.${BADGE_CLASS}`).forEach((b) => b.remove());
      if (wrapper.parentElement) {
        wrapper.parentElement.insertBefore(img, wrapper);
        wrapper.remove();
      }
      return;
    }
    img.style.opacity = "0.7";
    const id = Math.random().toString(36).slice(2);
    chrome.runtime.sendMessage({ type: "LADY_SIFT_CLASSIFY", url, id, threshold }, (res) => {
      img.style.opacity = "";
      if (chrome.runtime.lastError) {
        console.warn("[LadySift] classify error", chrome.runtime.lastError.message);
        return;
      }
      if (!res || res.error) {
        console.warn("[LadySift] classify failed", url, res?.error);
        return;
      }
      wrapAndBadge(img, res);
    });
  }

  function wrapAndBadge(img, { label }) {
    if (img.closest(`.${BADGE_CLASS}`)) return;
    const wrapper = document.createElement("span");
    wrapper.className = "lady-sift-wrapper";
    const parent = img.parentElement;
    if (!parent) return;
    // Avoid double-wrapping
    if (parent.classList.contains("lady-sift-wrapper")) {
      addBadge(parent, label);
      return;
    }
    img.before(wrapper);
    wrapper.appendChild(img);
    addBadge(wrapper, label);
  }

  function addBadge(wrapper, label) {
    wrapper.querySelectorAll(`.${BADGE_CLASS}`).forEach((b) => b.remove());
    const badge = document.createElement("span");
    badge.className = BADGE_CLASS;
    badge.dataset.label = label;
    // User-facing: AI Generated vs Real (no percent)
    const text = label === "fake" ? "AI Generated" : "Real";
    badge.textContent = text;
    badge.title = label === "fake" ? "AI-generated image" : "Real image";
    wrapper.appendChild(badge);
    wrapper.style.position = "relative";
  }

  // Observer for new images - just mark them clickable, don't auto-classify
  let scanTimer = null;
  function scan() {
    if (!enabled) return;
    markClickable();
  }
  const observer = new MutationObserver(() => {
    clearTimeout(scanTimer);
    scanTimer = setTimeout(scan, 800);
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
  if (document.readyState === "complete") scan();
  else window.addEventListener("load", scan);
})();
