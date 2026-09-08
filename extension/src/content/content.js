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
  `;
  document.documentElement.appendChild(style);

  chrome.storage.local.get({ enabled: true, threshold: 0.5 }, (v) => {
    enabled = v.enabled;
    threshold = v.threshold;
    if (enabled) scan();
  });
  chrome.storage.onChanged.addListener((changes) => {
    if (changes.threshold) threshold = changes.threshold.newValue;
    if (changes.enabled) {
      enabled = changes.enabled.newValue;
      if (enabled) scan();
      else clearBadges();
    }
  });

  function clearBadges() {
    document.querySelectorAll(`.${BADGE_CLASS}`).forEach((b) => b.remove());
    document.querySelectorAll(`[${PROCESSED_ATTR}]`).forEach((el) => el.removeAttribute(PROCESSED_ATTR));
  }

  function collectImages() {
    const imgs = [...document.querySelectorAll("img")]
      .filter((img) => !img.hasAttribute(PROCESSED_ATTR))
      .filter((img) => {
        const w = img.naturalWidth || img.width;
        const h = img.naturalHeight || img.height;
        return w >= MIN_SIZE && h >= MIN_SIZE;
      })
      .map((img) => ({ el: img, url: img.currentSrc || img.src }))
      .filter((x) => x.url && /^https?:|^blob:|^data:image/.test(x.url));

    // Dedup by url
    const seen = new Set();
    const unique = [];
    for (const item of imgs) {
      if (seen.has(item.url)) continue;
      seen.add(item.url);
      unique.push(item);
      if (unique.length >= MAX_PER_PAGE) break;
    }
    return unique;
  }

  function wrapAndBadge(img, { label, fakeProb }) {
    if (img.closest(`.${BADGE_CLASS}`)) return;
    const wrapper = document.createElement("span");
    wrapper.className = "lady-sift-wrapper";
    const parent = img.parentElement;
    if (!parent) return;
    // Avoid double-wrapping
    if (parent.classList.contains("lady-sift-wrapper")) {
      addBadge(parent, label, fakeProb);
      return;
    }
    img.before(wrapper);
    wrapper.appendChild(img);
    addBadge(wrapper, label, fakeProb);
  }

  function addBadge(wrapper, label, fakeProb) {
    wrapper.querySelectorAll(`.${BADGE_CLASS}`).forEach((b) => b.remove());
    const badge = document.createElement("span");
    badge.className = BADGE_CLASS;
    badge.dataset.label = label;
    badge.textContent = `${label} ${(fakeProb * 100).toFixed(0)}%`;
    wrapper.appendChild(badge);
    wrapper.style.position = "relative";
  }

  let scanTimer = null;
  function scan() {
    if (!enabled) return;
    const items = collectImages();
    if (!items.length) return;
    console.log(`[LadySift] classifying ${items.length} images (threshold ${threshold})`);
    for (const { el, url } of items) {
      el.setAttribute(PROCESSED_ATTR, "1");
      const id = Math.random().toString(36).slice(2);
      chrome.runtime.sendMessage({ type: "LADY_SIFT_CLASSIFY", url, id, threshold }, (res) => {
        if (chrome.runtime.lastError) {
          console.warn("[LadySift] classify error", chrome.runtime.lastError.message);
          return;
        }
        if (!res || res.error) {
          console.warn("[LadySift] classify failed", url, res?.error);
          return;
        }
        wrapAndBadge(el, res);
      });
    }
  }

  // Initial + observer
  const observer = new MutationObserver(() => {
    clearTimeout(scanTimer);
    scanTimer = setTimeout(scan, 800);
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });

  // Re-scan on load
  if (document.readyState === "complete") scan();
  else window.addEventListener("load", scan);
})();
