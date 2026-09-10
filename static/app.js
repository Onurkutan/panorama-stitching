// The page works against two back ends:
//  - the Python server in app.py (POST api/stitch), when it is running;
//  - the same Python modules executed inside the browser through Pyodide
//    (CPython + OpenCV compiled to WebAssembly) when there is no server,
//    which is how the GitHub Pages deployment runs.
// Both return the same shape: { metrics, files: { panorama, matches, ... } }.
// On failure both return { ok: false, error: "<English message>", code: "<code>" };
// the UI maps the code onto a localized message and falls back to the raw
// English text when a code is not recognized.

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js";
const PY_PACKAGE_DIR = "panorama_stitching";
const PY_MODULES = ["__init__.py", "errors.py", "features.py", "matching.py", "homography.py", "blending.py", "pipeline.py"];
// Longest input side used in the browser; SIFT in WebAssembly is a few times
// slower than native, and this keeps a pair under ~6 s.
const BROWSER_MAX_SIDE = 1400;
// Mirror of panorama_stitching.pipeline.EXAMPLES for the server-less mode (a
// test keeps the two lists in sync). Keep this one entry per line: a test
// regex-parses these fields.
const STATIC_EXAMPLES = [
  { id: "clock", title: "Clock tower", folder: "images/Clock", left: "sol1.jpg", right: "sag1.jpg" },
  { id: "school", title: "School yard", folder: "images/SchoolImage", left: "sol2.jpg", right: "sag2.jpg" },
  { id: "street", title: "Pont du Gard", folder: "images/test1", left: "s1.jpg", right: "s2.jpg" },
];

const LANG_STORAGE_KEY = "panorama-lang";

// ------------------------------------------------------------------ i18n ---

// Flat, dot-namespaced keys (no nested lookup) so t() stays a single object
// access. Error-code messages live in the same table under "error.<code>".
const STRINGS = {
  en: {
    "hero.eyebrow": "Computer Vision",
    "hero.intro": "A visual interface that runs SIFT, FLANN, RANSAC and feather blending on a single screen.",
    "hero.langAriaLabel": "Language",

    "controls.heading": "Input",
    "controls.modeAriaLabel": "Input mode",
    "mode.upload": "Your photos",
    "mode.example": "Demo pairs",

    "upload.leftLabel": "Left photo",
    "upload.rightLabel": "Right photo",
    "upload.chooseFile": "Choose a file",
    "upload.hint": "Tip: the left photo should be the one on the left, with about 30-50% overlap between the two shots.",
    "upload.swapAriaLabel": "Swap left/right",
    "upload.invalidFile": "Please choose an image file.",

    "example.datasetLabel": "Dataset",
    "example.leftCaption": "Left",
    "example.rightCaption": "Right",
    "example.leftAlt": "Left example photo",
    "example.rightAlt": "Right example photo",

    "actions.run": "Create panorama",
    "actions.running": "Running…",

    "status.ready": "Ready",
    "status.browserReady": 'No server detected: the computation runs in your browser. Click "Create panorama" to start.',
    "status.needBoth": "Please choose both the left and right photo.",
    "status.loadingRuntime": "Loading Python and OpenCV into your browser (first visit is about 20 MB)…",
    "status.fetchingExamples": "Fetching demo images…",
    "status.computingServer": "Computing SIFT points, matches and homography…",
    "status.computingBrowser": "Computing in your browser — this can take a few seconds and the page may pause…",
    "status.done": "Panorama ready: {width} x {height}px",
    "status.downscaleNote": " (inputs downscaled to {percent}%)",
    "status.browserNote": " · computed in your browser",
    "status.pyodideFailed": "Could not download Pyodide; check your connection.",
    "status.fetchFailed": "Could not fetch file: {url}",
    "status.exampleMissing": "The selected demo pair was not found.",
    "status.examplesFailed": "Could not fetch the demo pairs.",
    "status.stitchFailed": "Could not complete the operation.",

    "results.eyebrow": "Result",
    "results.heading": "Stitched panorama",
    "results.download": "Download panorama",
    "results.empty": "Your panorama will appear here",
    "results.detailsHeading": "Pipeline details",
    "results.tabsAriaLabel": "Intermediate images",
    "results.panoramaAlt": "Stitched panorama result",
    "results.detailAlt": "Intermediate pipeline image",

    "metrics.left": "Left keypoints",
    "metrics.right": "Right keypoints",
    "metrics.matches": "Good matches",
    "metrics.inliers": "RANSAC inliers",

    "tabs.matches": "Matches",
    "tabs.ransac": "RANSAC inliers",
    "tabs.leftKeypoints": "Left keypoints",
    "tabs.rightKeypoints": "Right keypoints",

    "error.image_unreadable": "The image could not be read.",
    "error.not_enough_features": "Not enough distinctive points were found in one of the photos.",
    "error.not_enough_matches": "Not enough matching points were found between the photos.",
    "error.homography_failed": "Could not compute a valid transform between the photos.",
    "error.degenerate_homography": "The computed transform was degenerate; try photos with more overlap.",
    "error.output_write_failed": "The result could not be saved.",
    "error.example_not_found": "The selected demo pair was not found.",
    "error.form_invalid": "The submitted form was invalid.",
    "error.upload_missing": "Please choose both the left and right photo.",
    "error.upload_not_image": "The uploaded file is not a readable image.",
    "error.upload_too_large": "The uploaded file is too large.",
    "error.unexpected": "An unexpected error occurred.",
  },
  tr: {
    "hero.eyebrow": "Bilgisayarlı Görü",
    "hero.intro": "SIFT, FLANN, RANSAC ve feather blending adımlarını tek ekranda çalıştıran görsel bir arayüz.",
    "hero.langAriaLabel": "Dil",

    "controls.heading": "Girdi",
    "controls.modeAriaLabel": "Girdi modu",
    "mode.upload": "Fotoğraflarınız",
    "mode.example": "Örnek çiftler",

    "upload.leftLabel": "Sol fotoğraf",
    "upload.rightLabel": "Sağ fotoğraf",
    "upload.chooseFile": "Dosya seç",
    "upload.hint": "İpucu: sol fotoğraf gerçekten solda çekilen olmalı; iki kare arasında yaklaşık %30-50 örtüşme olsun.",
    "upload.swapAriaLabel": "Sol ve sağı değiştir",
    "upload.invalidFile": "Lütfen bir görsel dosyası seçin.",

    "example.datasetLabel": "Veri seti",
    "example.leftCaption": "Sol",
    "example.rightCaption": "Sağ",
    "example.leftAlt": "Sol örnek görsel",
    "example.rightAlt": "Sağ örnek görsel",

    "actions.run": "Panorama Oluştur",
    "actions.running": "Çalışıyor…",

    "status.ready": "Hazır",
    "status.browserReady": 'Sunucu bulunamadı: işlem tarayıcınızda yapılır. Başlamak için "Panorama Oluştur"a tıklayın.',
    "status.needBoth": "Lütfen sol ve sağ fotoğrafı seçin.",
    "status.loadingRuntime": "Python ve OpenCV tarayıcınıza yükleniyor (ilk ziyarette yaklaşık 20 MB)…",
    "status.fetchingExamples": "Örnek görseller alınıyor…",
    "status.computingServer": "SIFT noktaları, eşleşmeler ve homografi hesaplanıyor…",
    "status.computingBrowser": "Tarayıcınızda hesaplanıyor — bu birkaç saniye sürebilir ve sayfa donmuş gibi görünebilir…",
    "status.done": "Panorama hazır: {width} x {height}px",
    "status.downscaleNote": " (girdiler %{percent} boyuta küçültüldü)",
    "status.browserNote": " · tarayıcıda hesaplandı",
    "status.pyodideFailed": "Pyodide indirilemedi; bağlantınızı kontrol edin.",
    "status.fetchFailed": "Dosya alınamadı: {url}",
    "status.exampleMissing": "Seçilen örnek çift bulunamadı.",
    "status.examplesFailed": "Örnek çiftler alınamadı.",
    "status.stitchFailed": "İşlem tamamlanamadı.",

    "results.eyebrow": "Sonuç",
    "results.heading": "Birleştirilmiş panorama",
    "results.download": "Panoramayı indir",
    "results.empty": "Panoramanız burada görünecek",
    "results.detailsHeading": "İşlem hattı ayrıntıları",
    "results.tabsAriaLabel": "Ara görseller",
    "results.panoramaAlt": "Birleştirilmiş panorama sonucu",
    "results.detailAlt": "Ara işlem görseli",

    "metrics.left": "Sol nokta",
    "metrics.right": "Sağ nokta",
    "metrics.matches": "İyi eşleşme",
    "metrics.inliers": "RANSAC iç nokta",

    "tabs.matches": "Eşleşmeler",
    "tabs.ransac": "RANSAC iç noktaları",
    "tabs.leftKeypoints": "Sol anahtar noktalar",
    "tabs.rightKeypoints": "Sağ anahtar noktalar",

    "error.image_unreadable": "Görsel okunamadı.",
    "error.not_enough_features": "Fotoğraflardan birinde yeterli sayıda belirgin nokta bulunamadı.",
    "error.not_enough_matches": "Fotoğraflar arasında yeterli eşleşme bulunamadı.",
    "error.homography_failed": "Fotoğraflar arasında geçerli bir dönüşüm hesaplanamadı.",
    "error.degenerate_homography": "Hesaplanan dönüşüm geçersiz (dejenere); daha fazla örtüşen fotoğraflar deneyin.",
    "error.output_write_failed": "Sonuç kaydedilemedi.",
    "error.example_not_found": "Seçilen örnek çift bulunamadı.",
    "error.form_invalid": "Gönderilen form geçersiz.",
    "error.upload_missing": "Lütfen sol ve sağ fotoğrafı seçin.",
    "error.upload_not_image": "Yüklenen dosya okunabilir bir görsel değil.",
    "error.upload_too_large": "Yüklenen dosya çok büyük.",
    "error.unexpected": "Beklenmeyen bir hata oluştu.",
  },
};

// Localized display titles for the fixed demo-pair ids; falls back to the
// backend-provided title (see STATIC_EXAMPLES / server) for unknown ids.
const EXAMPLE_TITLES = {
  en: { clock: "Clock tower", school: "School yard", street: "Pont du Gard" },
  tr: { clock: "Saat Kulesi", school: "Okul Bahçesi", street: "Pont du Gard" },
};

function detectInitialLang() {
  try {
    const stored = localStorage.getItem(LANG_STORAGE_KEY);
    if (stored === "en" || stored === "tr") return stored;
  } catch (error) {
    // Storage unavailable (private mode, disabled cookies, ...): ignore.
  }
  return (navigator.language || "").toLowerCase().startsWith("tr") ? "tr" : "en";
}

function t(key, params) {
  const template = STRINGS[state.lang][key] ?? STRINGS.en[key] ?? key;
  if (!params) return template;
  return template.replace(/\{(\w+)\}/g, (match, name) => (name in params ? String(params[name]) : match));
}

// An Error whose message is derived from an i18n key, so the status line can
// re-render it after a language switch instead of keeping a frozen string.
function localizedError(key, params) {
  const error = new Error(t(key, params));
  error.i18nKey = key;
  error.params = params;
  return error;
}

function localizedExampleTitle(example) {
  return EXAMPLE_TITLES[state.lang]?.[example.id] || example.title;
}

// ----------------------------------------------------------------- state ---

const state = {
  lang: detectInitialLang(),
  mode: "upload",
  examples: [],
  resultFiles: {},
  lastMetrics: null,
  backend: null,
  running: false,
  leftFile: null,
  rightFile: null,
  leftPreviewUrl: null,
  rightPreviewUrl: null,
  statusRenderer: () => t("status.ready"),
  statusIsError: false,
};

const els = {
  langButtons: document.querySelectorAll(".lang-btn"),
  modes: document.querySelectorAll(".mode"),
  examplePanel: document.querySelector("#examplePanel"),
  uploadPanel: document.querySelector("#uploadPanel"),
  exampleSelect: document.querySelector("#exampleSelect"),
  exampleLeft: document.querySelector("#exampleLeft"),
  exampleRight: document.querySelector("#exampleRight"),
  leftZone: document.querySelector("#leftZone"),
  rightZone: document.querySelector("#rightZone"),
  leftImage: document.querySelector("#leftImage"),
  rightImage: document.querySelector("#rightImage"),
  leftPreview: document.querySelector("#leftPreview"),
  rightPreview: document.querySelector("#rightPreview"),
  leftFileName: document.querySelector("#leftFileName"),
  rightFileName: document.querySelector("#rightFileName"),
  swapButton: document.querySelector("#swapButton"),
  runButton: document.querySelector("#runButton"),
  runButtonLabel: document.querySelector("#runButtonLabel"),
  progressBar: document.querySelector("#progressBar"),
  status: document.querySelector("#status"),
  panoramaEmpty: document.querySelector("#panoramaEmpty"),
  panoramaImage: document.querySelector("#panoramaImage"),
  detailImage: document.querySelector("#detailImage"),
  tabs: document.querySelectorAll(".tab"),
  downloadButton: document.querySelector("#downloadButton"),
  mLeft: document.querySelector("#mLeft"),
  mRight: document.querySelector("#mRight"),
  mMatches: document.querySelector("#mMatches"),
  mInliers: document.querySelector("#mInliers"),
};

// A status renderer is a zero-arg function that calls t() itself, so it can
// be re-evaluated after a language switch instead of showing frozen text.
function setStatus(renderer, isError = false) {
  state.statusRenderer = renderer;
  state.statusIsError = isError;
  renderStatus();
}

function renderStatus() {
  els.status.textContent = state.statusRenderer();
  els.status.classList.toggle("error", state.statusIsError);
}

function resolveErrorMessage(error) {
  if (error && error.i18nKey) return t(error.i18nKey, error.params);
  if (error && error.code && STRINGS.en[`error.${error.code}`]) {
    return t(`error.${error.code}`);
  }
  return (error && error.message) || t("status.stitchFailed");
}

// Let the browser paint the status line before a long synchronous job.
function nextPaint() {
  return new Promise((resolve) => setTimeout(resolve, 30));
}

function withCacheBuster(url) {
  return url.startsWith("blob:") ? url : `${url}?t=${Date.now()}`;
}

function buildDoneMessage(metrics, backendName) {
  let message = t("status.done", { width: metrics.panoramaWidth, height: metrics.panoramaHeight });
  if (metrics.inputScale && metrics.inputScale < 1) {
    message += t("status.downscaleNote", { percent: Math.round(metrics.inputScale * 100) });
  }
  if (backendName === "browser") {
    message += t("status.browserNote");
  }
  return message;
}

// ---------------------------------------------------------------- server ---

const serverBackend = {
  name: "server",

  async listExamples() {
    const response = await fetch("api/examples");
    if (!response.ok) throw localizedError("status.examplesFailed");
    return (await response.json()).examples;
  },

  async stitch(input) {
    const formData = new FormData();
    formData.append("mode", input.mode);
    if (input.mode === "example") {
      formData.append("example", input.exampleId);
    } else {
      formData.append("leftImage", input.leftFile);
      formData.append("rightImage", input.rightFile);
    }
    const response = await fetch("api/stitch", { method: "POST", body: formData });
    const data = await response.json();
    if (!data.ok) {
      const error = new Error(data.error || "Stitching failed.");
      error.code = data.code || "unexpected";
      throw error;
    }
    return { metrics: data.metrics, files: data.files };
  },
};

// --------------------------------------------------------------- browser ---

function loadScript(url) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = url;
    script.onload = resolve;
    script.onerror = () => reject(localizedError("status.pyodideFailed"));
    document.head.appendChild(script);
  });
}

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw localizedError("status.fetchFailed", { url });
  return new Uint8Array(await response.arrayBuffer());
}

const browserBackend = {
  name: "browser",
  pyodide: null,
  ready: null,
  fetched: new Set(),

  async listExamples() {
    return STATIC_EXAMPLES.map((example) => ({
      id: example.id,
      title: example.title,
      left: `${example.folder}/${example.left}`,
      right: `${example.folder}/${example.right}`,
    }));
  },

  async ensureReady(onStatus) {
    if (!this.ready) {
      this.ready = (async () => {
        onStatus({ key: "status.loadingRuntime" });
        await loadScript(PYODIDE_URL);
        const pyodide = await window.loadPyodide();
        await pyodide.loadPackage(["numpy", "opencv-python"]);
        pyodide.FS.mkdirTree("/app/out");
        pyodide.FS.mkdirTree("/app/in");
        pyodide.FS.mkdirTree(`/app/${PY_PACKAGE_DIR}`);
        for (const name of PY_MODULES) {
          pyodide.FS.writeFile(`/app/${PY_PACKAGE_DIR}/${name}`, await fetchBytes(`${PY_PACKAGE_DIR}/${name}`));
        }
        pyodide.runPython("import sys\nsys.path.insert(0, '/app')");
        this.pyodide = pyodide;
      })().catch((error) => {
        this.ready = null;
        throw error;
      });
    }
    await this.ready;
    return this.pyodide;
  },

  async fetchIntoFS(pyodide, url, path) {
    if (this.fetched.has(path)) return;
    pyodide.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
    pyodide.FS.writeFile(path, await fetchBytes(url));
    this.fetched.add(path);
  },

  removeTree(pyodide, dir) {
    let entries;
    try {
      entries = pyodide.FS.readdir(dir).filter((name) => name !== "." && name !== "..");
    } catch (error) {
      return;
    }
    for (const name of entries) pyodide.FS.unlink(`${dir}/${name}`);
    pyodide.FS.rmdir(dir);
  },

  async stitch(input, onStatus) {
    const pyodide = await this.ensureReady(onStatus);
    const job = Date.now().toString(36);
    const inDir = `/app/in/${job}`;
    const outDir = `/app/out/${job}`;
    let left;
    let right;

    if (input.mode === "example") {
      const example = STATIC_EXAMPLES.find((item) => item.id === input.exampleId);
      if (!example) throw localizedError("status.exampleMissing");
      left = `/app/${example.folder}/${example.left}`;
      right = `/app/${example.folder}/${example.right}`;
      onStatus({ key: "status.fetchingExamples" });
      await this.fetchIntoFS(pyodide, `${example.folder}/${example.left}`, left);
      await this.fetchIntoFS(pyodide, `${example.folder}/${example.right}`, right);
    } else {
      pyodide.FS.mkdirTree(inDir);
      left = `${inDir}/left${extensionOf(input.leftFile.name)}`;
      right = `${inDir}/right${extensionOf(input.rightFile.name)}`;
      pyodide.FS.writeFile(left, new Uint8Array(await input.leftFile.arrayBuffer()));
      pyodide.FS.writeFile(right, new Uint8Array(await input.rightFile.arrayBuffer()));
    }

    onStatus({ key: "status.computingBrowser" });
    await nextPaint();

    // Same call the server makes; PanoramaError becomes {ok: false, error, code}.
    const script = `
import json
from panorama_stitching.pipeline import PanoramaError, stitch_pair
try:
    result = stitch_pair(${JSON.stringify(left)}, ${JSON.stringify(right)},
                         ${JSON.stringify(outDir)}, max_side=${BROWSER_MAX_SIDE})
    result["ok"] = True
except PanoramaError as exc:
    result = {"ok": False, "error": str(exc), "code": getattr(exc, "code", "unexpected")}
json.dumps(result)
`;
    const result = JSON.parse(pyodide.runPython(script));
    if (!result.ok) {
      this.removeTree(pyodide, outDir);
      this.removeTree(pyodide, inDir);
      const error = new Error(result.error || "Stitching failed.");
      error.code = result.code || "unexpected";
      throw error;
    }

    const files = {};
    for (const [key, name] of Object.entries(result.files)) {
      const bytes = pyodide.FS.readFile(`${outDir}/${name}`);
      files[key] = URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }));
    }
    // Outputs now live in blob URLs; free the in-memory file system again.
    this.removeTree(pyodide, outDir);
    this.removeTree(pyodide, inDir);
    return { metrics: result.metrics, files };
  },
};

function extensionOf(fileName) {
  const match = /\.[a-z0-9]+$/i.exec(fileName || "");
  return match ? match[0].toLowerCase() : ".jpg";
}

async function pickBackend() {
  try {
    const response = await fetch("api/examples", { cache: "no-store" });
    const type = response.headers.get("content-type") || "";
    if (response.ok && type.includes("json")) return serverBackend;
  } catch (error) {
    // No server: fall through to the in-browser pipeline.
  }
  return browserBackend;
}

// -------------------------------------------------------------------- UI ---

function setLang(lang) {
  if (lang !== "en" && lang !== "tr") return;
  state.lang = lang;
  try {
    localStorage.setItem(LANG_STORAGE_KEY, lang);
  } catch (error) {
    // Storage unavailable: the choice just won't persist across reloads.
  }
  renderAll();
}

function applyStaticTranslations() {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-aria-label]").forEach((el) => {
    el.setAttribute("aria-label", t(el.dataset.i18nAriaLabel));
  });
  document.querySelectorAll("[data-i18n-alt]").forEach((el) => {
    el.alt = t(el.dataset.i18nAlt);
  });
}

function renderLangToggle() {
  els.langButtons.forEach((button) => {
    const active = button.dataset.lang === state.lang;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

function renderRunButton() {
  els.runButtonLabel.textContent = t(state.running ? "actions.running" : "actions.run");
  els.runButton.disabled = state.running;
  els.progressBar.hidden = !state.running;
}

function renderUploadPreview(side) {
  const file = state[`${side}File`];
  const previewUrl = state[`${side}PreviewUrl`];
  els[`${side}FileName`].textContent = file ? file.name : t("upload.chooseFile");
  const imgEl = els[`${side}Preview`];
  if (previewUrl) {
    imgEl.src = previewUrl;
    imgEl.hidden = false;
  } else {
    imgEl.hidden = true;
    imgEl.removeAttribute("src");
  }
}

function renderExampleOptions() {
  const previousValue = els.exampleSelect.value;
  els.exampleSelect.innerHTML = state.examples
    .map((example) => `<option value="${example.id}">${localizedExampleTitle(example)}</option>`)
    .join("");
  const hasPrevious = state.examples.some((example) => example.id === previousValue);
  els.exampleSelect.value = hasPrevious ? previousValue : state.examples[0]?.id || "";
  refreshExamplePreview();
}

function renderAll() {
  document.documentElement.lang = state.lang;
  applyStaticTranslations();
  renderLangToggle();
  renderRunButton();
  renderUploadPreview("left");
  renderUploadPreview("right");
  renderExampleOptions();
  if (state.lastMetrics) setMetrics(state.lastMetrics);
  renderStatus();
}

function setMode(mode) {
  state.mode = mode;
  els.modes.forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  els.uploadPanel.classList.toggle("hidden", mode !== "upload");
  els.examplePanel.classList.toggle("hidden", mode !== "example");
}

function refreshExamplePreview() {
  const example = state.examples.find((item) => item.id === els.exampleSelect.value);
  if (!example) return;
  els.exampleLeft.src = example.left;
  els.exampleRight.src = example.right;
}

function setUploadFile(side, file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setStatus(() => t("upload.invalidFile"), true);
    return;
  }
  const urlKey = `${side}PreviewUrl`;
  if (state[urlKey]) URL.revokeObjectURL(state[urlKey]);
  state[`${side}File`] = file;
  state[urlKey] = URL.createObjectURL(file);
  renderUploadPreview(side);
}

function swapUploads() {
  const leftFile = state.leftFile;
  const leftPreviewUrl = state.leftPreviewUrl;
  state.leftFile = state.rightFile;
  state.leftPreviewUrl = state.rightPreviewUrl;
  state.rightFile = leftFile;
  state.rightPreviewUrl = leftPreviewUrl;
  renderUploadPreview("left");
  renderUploadPreview("right");
}

function wireDropZone(zoneEl, side) {
  ["dragenter", "dragover"].forEach((eventName) => {
    zoneEl.addEventListener(eventName, (event) => {
      event.preventDefault();
      zoneEl.classList.add("drag-over");
    });
  });
  ["dragleave", "dragend"].forEach((eventName) => {
    zoneEl.addEventListener(eventName, () => zoneEl.classList.remove("drag-over"));
  });
  zoneEl.addEventListener("drop", (event) => {
    event.preventDefault();
    zoneEl.classList.remove("drag-over");
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    setUploadFile(side, file);
  });
}

function setMetrics(metrics) {
  const locale = state.lang === "tr" ? "tr-TR" : "en-US";
  els.mLeft.textContent = metrics.leftKeypoints.toLocaleString(locale);
  els.mRight.textContent = metrics.rightKeypoints.toLocaleString(locale);
  els.mMatches.textContent = metrics.goodMatches.toLocaleString(locale);
  els.mInliers.textContent = metrics.inliers.toLocaleString(locale);
}

function selectDetail(target) {
  const src = state.resultFiles[target];
  if (!src) return;
  els.tabs.forEach((tab) => {
    const active = tab.dataset.target === target;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  els.detailImage.src = withCacheBuster(src);
  els.detailImage.hidden = false;
}

function setResult(data) {
  // Release blob URLs of the previous run before dropping the references.
  Object.values(state.resultFiles)
    .filter((url) => url.startsWith("blob:"))
    .forEach((url) => URL.revokeObjectURL(url));
  state.resultFiles = data.files;
  state.lastMetrics = data.metrics;
  els.panoramaEmpty.hidden = true;
  els.panoramaImage.hidden = false;
  els.panoramaImage.src = withCacheBuster(data.files.panorama);
  els.downloadButton.href = data.files.panorama;
  els.downloadButton.classList.remove("disabled");
  setMetrics(data.metrics);
  selectDetail("matches");
}

async function loadExamples() {
  state.examples = await state.backend.listExamples();
  renderExampleOptions();
}

async function runStitching() {
  if (state.running) return;

  const input = { mode: state.mode };
  if (state.mode === "example") {
    input.exampleId = els.exampleSelect.value;
  } else {
    if (!state.leftFile || !state.rightFile) {
      setStatus(() => t("status.needBoth"), true);
      return;
    }
    input.leftFile = state.leftFile;
    input.rightFile = state.rightFile;
  }

  state.running = true;
  renderRunButton();
  if (state.backend.name === "server") {
    setStatus(() => t("status.computingServer"));
  }

  try {
    const data = await state.backend.stitch(input, (entry) => setStatus(() => t(entry.key, entry.params)));
    setResult(data);
    setStatus(() => buildDoneMessage(data.metrics, state.backend.name));
  } catch (error) {
    setStatus(() => resolveErrorMessage(error), true);
  } finally {
    state.running = false;
    renderRunButton();
  }
}

els.langButtons.forEach((button) => {
  button.addEventListener("click", () => setLang(button.dataset.lang));
});

els.modes.forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

els.exampleSelect.addEventListener("change", refreshExamplePreview);
els.runButton.addEventListener("click", runStitching);
els.tabs.forEach((tab) => tab.addEventListener("click", () => selectDetail(tab.dataset.target)));
els.swapButton.addEventListener("click", swapUploads);

els.leftImage.addEventListener("change", () => setUploadFile("left", els.leftImage.files[0]));
els.rightImage.addEventListener("change", () => setUploadFile("right", els.rightImage.files[0]));
wireDropZone(els.leftZone, "left");
wireDropZone(els.rightZone, "right");

async function boot() {
  renderAll();
  setMode(state.mode);
  state.backend = await pickBackend();
  await loadExamples();
  if (state.backend.name === "server") {
    setStatus(() => t("status.ready"));
  } else {
    setStatus(() => t("status.browserReady"));
  }
}

boot().catch((error) => setStatus(() => resolveErrorMessage(error), true));
