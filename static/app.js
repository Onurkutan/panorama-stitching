// The page works against two back ends:
//  - the Python server in app.py (POST api/stitch), when it is running;
//  - the same Python modules executed inside a Web Worker through Pyodide
//    (CPython + OpenCV compiled to WebAssembly) when there is no server,
//    which is how the GitHub Pages deployment runs. The worker (static/
//    worker.js) keeps Pyodide off the main thread so the page never freezes;
//    see that file for the runtime/package/module loading it owns.
// Both return the same shape: { metrics, files: { panorama, keypoints, ... } }.
// On failure both return { ok: false, error: "<English message>", code: "<code>", details }
// the UI maps the code onto a localized message and falls back to the raw
// English text when a code is not recognized.

// Longest input side used in the browser; SIFT in WebAssembly is a few times
// slower than native, and this keeps a run reasonably fast.
const BROWSER_MAX_SIDE = 1400;
// "Your photos" accepts this many images, in any order; the pipeline itself
// recovers the left-to-right arrangement.
// Photos larger than this on the long side are shrunk in the browser before
// they are sent anywhere: the pipeline downscales to this size anyway, and a
// 24-megapixel phone photo would otherwise cost 70 MB of memory per image to
// decode in the worker (or a 40 MB upload to the server). Decoding through
// createImageBitmap also applies the EXIF orientation.
const UPLOAD_MAX_SIDE = 2400;
const MIN_PHOTOS = 2;
const MAX_PHOTOS = 6;
// Mirror of panorama_stitching.pipeline.EXAMPLES for the server-less mode (a
// test keeps the two lists in sync). Keep this one entry per line: a test
// regex-parses these fields.
const STATIC_EXAMPLES = [
  { id: "clock", title: "Clock tower", folder: "images/Clock", files: ["sol1.jpg", "sag1.jpg"] },
  { id: "school", title: "School yard", folder: "images/SchoolImage", files: ["sol2.jpg", "sag2.jpg"] },
  { id: "street", title: "Pont du Gard", folder: "images/test1", files: ["s1.jpg", "s2.jpg"] },
  { id: "balcony", title: "Balcony sweep", folder: "images/BalconySweep", files: ["photo_1.jpg", "photo_2.jpg", "photo_3.jpg", "photo_4.jpg", "photo_5.jpg", "photo_6.jpg"] },
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
    "mode.example": "Demo sets",

    "upload.zoneTitle": "Add photos",
    "upload.zoneCta": "Click or drag photos here",
    "upload.hint": "Add 2 to 6 overlapping photos in any order; the arrangement is detected automatically.",
    "upload.counter": "{count} / {max} photos",
    "upload.removeAriaLabel": "Remove photo {index}",
    "upload.invalidFile": "Please choose image files.",

    "example.datasetLabel": "Dataset",
    "example.photoCaption": "Photo {index}",
    "example.photoAlt": "Demo photo {index}",

    "actions.run": "Create panorama",
    "actions.running": "Running…",
    "actions.cancel": "Cancel",

    "status.ready": "Ready",
    "status.browserReady": 'No server detected: the computation runs in your browser. Click "Create panorama" to start.',
    "status.needPhotos": "Please add between 2 and 6 photos.",
    "status.preparingPhotos": "Preparing the photos…",
    "status.projectionNote": " · cylindrical projection",
    "status.computingServer": "Computing SIFT points, matches and homography…",
    "status.done": "Panorama ready: {width} x {height}px",
    "status.downscaleNote": " (inputs downscaled to {percent}%)",
    "status.browserNote": " · computed in your browser",
    "status.cancelled": "Cancelled.",
    "status.exampleMissing": "The selected demo pair was not found.",
    "status.examplesFailed": "Could not fetch the demo pairs.",
    "status.stitchFailed": "Could not complete the operation.",

    // Reported while the in-browser (Web Worker + Pyodide) pipeline runs, in
    // the fixed order runtime -> packages -> modules -> inputs -> features ->
    // matching -> homography -> blending -> saving. The first four are the
    // worker's own setup steps (1-4 of 4); the last five report the step and
    // total the Python pipeline computed for the whole photo set (see
    // static/worker.js). The first stages are skipped once the runtime is
    // already warm.
    "stage.runtime": "Loading the Python runtime ({step}/{total})…",
    "stage.packages": "Loading NumPy and OpenCV ({step}/{total})…",
    "stage.modules": "Loading the panorama pipeline ({step}/{total})…",
    "stage.inputs": "Preparing the input photos ({step}/{total})…",
    "stage.features": "Detecting SIFT features ({step}/{total})…",
    "stage.matching": "Matching photos ({step}/{total})…",
    "stage.homography": "Estimating homography with RANSAC ({step}/{total})…",
    "stage.blending": "Blending the panorama ({step}/{total})…",
    "stage.saving": "Saving the results ({step}/{total})…",

    "results.eyebrow": "Result",
    "results.heading": "Stitched panorama",
    "results.download": "Download panorama",
    "results.empty": "Your panorama will appear here",
    "results.detailsHeading": "Pipeline details",
    "results.tabsAriaLabel": "Intermediate images",
    "results.panoramaAlt": "Stitched panorama result",
    "results.detailAlt": "Intermediate pipeline image",

    "metrics.keypoints": "Total keypoints",
    "metrics.pairs": "Pairs used",
    "metrics.pairsValue": "{used} of {evaluated} evaluated",
    "metrics.inliers": "RANSAC inliers",
    "metrics.order": "Detected order",

    "tabs.matchesPair": "Matches {i}+{j}",
    "tabs.ransacPair": "RANSAC {i}+{j}",
    "tabs.keypointsImage": "Keypoints {i}",

    "error.image_unreadable": "The image could not be read.",
    "error.not_enough_features": "Not enough distinctive points were found in one of the photos.",
    "error.not_enough_matches": "Not enough matching points were found between the photos.",
    "error.homography_failed": "Could not compute a valid transform between the photos.",
    "error.degenerate_homography": "The computed transform was degenerate; try photos with more overlap.",
    "error.output_write_failed": "The result could not be saved.",
    "error.example_not_found": "The selected demo pair was not found.",
    "error.form_invalid": "The submitted form was invalid.",
    "error.upload_missing": "Please add at least 2 photos.",
    "error.upload_not_image": "The uploaded file is not a readable image.",
    "error.upload_too_large": "The uploaded file is too large.",
    "error.too_few_images": "Please add at least 2 photos.",
    "error.too_many_images": "You can add at most 6 photos.",
    "error.image_not_connected": "One of the photos does not overlap with the others.",
    "error.pyodide_failed": "Could not load Python and OpenCV in your browser; check your connection.",
    "error.fetch_failed": "Could not fetch one of the required files; check your connection.",
    "error.unexpected": "An unexpected error occurred.",
    "error.heic_unsupported":
      "HEIC photos could not be decoded here. Export them as JPEG (iPhone: Settings > Camera > Formats > Most Compatible), use Safari, or install the optional HEIC support on the server.",
    "error.pairSuffix": " (photos {i} and {j})",
    "error.imageSuffix": " (photo {i})",
  },
  tr: {
    "hero.eyebrow": "Bilgisayarlı Görü",
    "hero.intro": "SIFT, FLANN, RANSAC ve feather blending adımlarını tek ekranda çalıştıran görsel bir arayüz.",
    "hero.langAriaLabel": "Dil",

    "controls.heading": "Girdi",
    "controls.modeAriaLabel": "Girdi modu",
    "mode.upload": "Fotoğraflarınız",
    "mode.example": "Örnek setler",

    "upload.zoneTitle": "Fotoğraf ekleyin",
    "upload.zoneCta": "Tıklayın veya fotoğrafları buraya sürükleyin",
    "upload.hint": "Herhangi bir sırada, 2 ile 6 arasında örtüşen fotoğraf ekleyin; düzen otomatik olarak tespit edilir.",
    "upload.counter": "{count} / {max} fotoğraf",
    "upload.removeAriaLabel": "{index}. fotoğrafı kaldır",
    "upload.invalidFile": "Lütfen görsel dosyaları seçin.",

    "example.datasetLabel": "Veri seti",
    "example.photoCaption": "Fotoğraf {index}",
    "example.photoAlt": "Örnek fotoğraf {index}",

    "actions.run": "Panorama Oluştur",
    "actions.running": "Çalışıyor…",
    "actions.cancel": "İptal",

    "status.ready": "Hazır",
    "status.browserReady": 'Sunucu bulunamadı: işlem tarayıcınızda yapılır. Başlamak için "Panorama Oluştur"a tıklayın.',
    "status.needPhotos": "Lütfen 2 ile 6 arasında fotoğraf ekleyin.",
    "status.preparingPhotos": "Fotoğraflar hazırlanıyor…",
    "status.projectionNote": " · silindirik projeksiyon",
    "status.computingServer": "SIFT noktaları, eşleşmeler ve homografi hesaplanıyor…",
    "status.done": "Panorama hazır: {width} x {height}px",
    "status.downscaleNote": " (girdiler %{percent} boyuta küçültüldü)",
    "status.browserNote": " · tarayıcıda hesaplandı",
    "status.cancelled": "İptal edildi.",
    "status.exampleMissing": "Seçilen örnek çift bulunamadı.",
    "status.examplesFailed": "Örnek çiftler alınamadı.",
    "status.stitchFailed": "İşlem tamamlanamadı.",

    "stage.runtime": "Python çalışma zamanı yükleniyor ({step}/{total})…",
    "stage.packages": "NumPy ve OpenCV yükleniyor ({step}/{total})…",
    "stage.modules": "Panorama işlem hattı yükleniyor ({step}/{total})…",
    "stage.inputs": "Girdi fotoğrafları hazırlanıyor ({step}/{total})…",
    "stage.features": "SIFT özellikleri tespit ediliyor ({step}/{total})…",
    "stage.matching": "Fotoğraflar eşleştiriliyor ({step}/{total})…",
    "stage.homography": "RANSAC ile homografi hesaplanıyor ({step}/{total})…",
    "stage.blending": "Panorama birleştiriliyor ({step}/{total})…",
    "stage.saving": "Sonuçlar kaydediliyor ({step}/{total})…",

    "results.eyebrow": "Sonuç",
    "results.heading": "Birleştirilmiş panorama",
    "results.download": "Panoramayı indir",
    "results.empty": "Panoramanız burada görünecek",
    "results.detailsHeading": "İşlem hattı ayrıntıları",
    "results.tabsAriaLabel": "Ara görseller",
    "results.panoramaAlt": "Birleştirilmiş panorama sonucu",
    "results.detailAlt": "Ara işlem görseli",

    "metrics.keypoints": "Toplam anahtar nokta",
    "metrics.pairs": "Kullanılan çift",
    "metrics.pairsValue": "{evaluated} çiftten {used} tanesi kullanıldı",
    "metrics.inliers": "RANSAC iç nokta",
    "metrics.order": "Tespit edilen sıra",

    "tabs.matchesPair": "Eşleşmeler {i}+{j}",
    "tabs.ransacPair": "RANSAC {i}+{j}",
    "tabs.keypointsImage": "Anahtar noktalar {i}",

    "error.image_unreadable": "Görsel okunamadı.",
    "error.not_enough_features": "Fotoğraflardan birinde yeterli sayıda belirgin nokta bulunamadı.",
    "error.not_enough_matches": "Fotoğraflar arasında yeterli eşleşme bulunamadı.",
    "error.homography_failed": "Fotoğraflar arasında geçerli bir dönüşüm hesaplanamadı.",
    "error.degenerate_homography": "Hesaplanan dönüşüm geçersiz (dejenere); daha fazla örtüşen fotoğraflar deneyin.",
    "error.output_write_failed": "Sonuç kaydedilemedi.",
    "error.example_not_found": "Seçilen örnek çift bulunamadı.",
    "error.form_invalid": "Gönderilen form geçersiz.",
    "error.upload_missing": "Lütfen en az 2 fotoğraf ekleyin.",
    "error.upload_not_image": "Yüklenen dosya okunabilir bir görsel değil.",
    "error.upload_too_large": "Yüklenen dosya çok büyük.",
    "error.too_few_images": "Lütfen en az 2 fotoğraf ekleyin.",
    "error.too_many_images": "En fazla 6 fotoğraf ekleyebilirsiniz.",
    "error.image_not_connected": "Fotoğraflardan biri diğerleriyle örtüşmüyor.",
    "error.pyodide_failed": "Python ve OpenCV tarayıcınıza yüklenemedi; bağlantınızı kontrol edin.",
    "error.fetch_failed": "Gerekli dosyalardan biri alınamadı; bağlantınızı kontrol edin.",
    "error.unexpected": "Beklenmeyen bir hata oluştu.",
    "error.heic_unsupported":
      "HEIC fotoğraflar burada çözülemedi. JPEG olarak dışa aktarın (iPhone: Ayarlar > Kamera > Biçimler > En Uyumlu), Safari kullanın veya sunucuya isteğe bağlı HEIC desteğini kurun.",
    "error.pairSuffix": " (fotoğraf {i} ve {j})",
    "error.imageSuffix": " (fotoğraf {i})",
  },
};

// Localized display titles for the fixed demo-pair ids; falls back to the
// backend-provided title (see STATIC_EXAMPLES / server) for unknown ids.
const EXAMPLE_TITLES = {
  en: { clock: "Clock tower", school: "School yard", street: "Pont du Gard", balcony: "Balcony sweep (6 photos)" },
  tr: { clock: "Saat Kulesi", school: "Okul Bahçesi", street: "Pont du Gard", balcony: "Balkon taraması (6 fotoğraf)" },
};

function detectInitialLang() {
  try {
    const stored = localStorage.getItem(LANG_STORAGE_KEY);
    if (stored === "en" || stored === "tr") return stored;
  } catch (error) {
    // Storage unavailable (private mode, disabled cookies, ...): ignore.
  }
  return "en";
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
  activeDetailTarget: null,
  lastMetrics: null,
  backend: null,
  running: false,
  photos: [], // ordered list of { file, previewUrl }; order is just the
  // upload order shown in the grid, not a left-to-right claim - the
  // pipeline figures the arrangement out on its own.
  statusRenderer: () => t("status.ready"),
  statusIsError: false,
};

const els = {
  langButtons: document.querySelectorAll(".lang-btn"),
  modes: document.querySelectorAll(".mode"),
  examplePanel: document.querySelector("#examplePanel"),
  uploadPanel: document.querySelector("#uploadPanel"),
  exampleSelect: document.querySelector("#exampleSelect"),
  examplePreview: document.querySelector("#examplePreview"),
  dropZone: document.querySelector("#dropZone"),
  photosInput: document.querySelector("#photosInput"),
  photoCounter: document.querySelector("#photoCounter"),
  photoGrid: document.querySelector("#photoGrid"),
  runButton: document.querySelector("#runButton"),
  runButtonLabel: document.querySelector("#runButtonLabel"),
  cancelButton: document.querySelector("#cancelButton"),
  progressBar: document.querySelector("#progressBar"),
  status: document.querySelector("#status"),
  panoramaEmpty: document.querySelector("#panoramaEmpty"),
  panoramaImage: document.querySelector("#panoramaImage"),
  detailTabs: document.querySelector("#detailTabs"),
  detailImage: document.querySelector("#detailImage"),
  downloadButton: document.querySelector("#downloadButton"),
  mKeypoints: document.querySelector("#mKeypoints"),
  mPairs: document.querySelector("#mPairs"),
  mInliers: document.querySelector("#mInliers"),
  mOrder: document.querySelector("#mOrder"),
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
  let message;
  if (error && error.i18nKey) {
    message = t(error.i18nKey, error.params);
  } else if (error && error.code && STRINGS.en[`error.${error.code}`]) {
    message = t(`error.${error.code}`);
  } else {
    message = (error && error.message) || t("status.stitchFailed");
  }
  const details = error && error.details;
  if (details && Array.isArray(details.pair)) {
    message += t("error.pairSuffix", { i: details.pair[0], j: details.pair[1] });
  } else if (details && details.image != null) {
    message += t("error.imageSuffix", { i: details.image });
  }
  return message;
}

function withCacheBuster(url) {
  return url.startsWith("blob:") ? url : `${url}?t=${Date.now()}`;
}

function buildDoneMessage(metrics, backendName) {
  let message = t("status.done", { width: metrics.panoramaWidth, height: metrics.panoramaHeight });
  if (metrics.inputScale && metrics.inputScale < 1) {
    message += t("status.downscaleNote", { percent: Math.round(metrics.inputScale * 100) });
  }
  if (metrics.projection === "cylindrical") {
    message += t("status.projectionNote");
  }
  if (backendName === "browser") {
    message += t("status.browserNote");
  }
  return message;
}

// Shrinks a photo to UPLOAD_MAX_SIDE on its long side (JPEG, EXIF orientation
// applied). Anything the browser cannot decode (for example HEIC outside
// Safari) is passed through unchanged and left to the pipeline to reject.
function looksLikeHeic(file) {
  const type = (file.type || "").toLowerCase();
  return type === "image/heic" || type === "image/heif" || /\.(heic|heif|hif)$/i.test(file.name || "");
}

// canDefer: whether an undecodable file may be passed through untouched. The
// server may still decode HEIC with its optional decoder; the in-browser
// pipeline never can, so there the failure is reported right away.
async function shrinkForUpload(file, canDefer) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch (error) {
    if (!canDefer && looksLikeHeic(file)) {
      const failure = new Error("HEIC photo cannot be decoded in this browser.");
      failure.code = "heic_unsupported";
      throw failure;
    }
    return file;
  }
  try {
    const longest = Math.max(bitmap.width, bitmap.height);
    if (longest <= UPLOAD_MAX_SIDE) return file;
    const scale = UPLOAD_MAX_SIDE / longest;
    const width = Math.round(bitmap.width * scale);
    const height = Math.round(bitmap.height * scale);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").drawImage(bitmap, 0, 0, width, height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    if (!blob) return file;
    const name = file.name.replace(/\.[^.]+$/, "") + ".jpg";
    return new File([blob], name, { type: "image/jpeg" });
  } finally {
    bitmap.close();
  }
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
      input.files.forEach((file) => formData.append("images", file));
    }
    const response = await fetch("api/stitch", { method: "POST", body: formData });
    const data = await response.json();
    if (!data.ok) {
      const error = new Error(data.error || "Stitching failed.");
      error.code = data.code || "unexpected";
      error.details = data.details || null;
      throw error;
    }
    return { metrics: data.metrics, files: data.files };
  },
};

// --------------------------------------------------------------- browser ---

// Runs the same panorama_stitching pipeline off the main thread, inside
// static/worker.js, so a several-second SIFT/RANSAC/blend pass never freezes
// the page. The worker owns Pyodide entirely (loading it is what the
// "runtime"/"packages"/"modules" status stages refer to); this object just
// starts the worker lazily, keeps it alive across runs, and turns its
// message protocol into the same { metrics, files } / thrown-Error(code)
// shape serverBackend uses.
const browserBackend = {
  name: "browser",
  worker: null,
  workerReady: null,
  workerResolveReady: null,
  workerRejectReady: null,
  pending: null,
  onProgress: null,
  jobCounter: 0,

  async listExamples() {
    return STATIC_EXAMPLES.map((example) => ({
      id: example.id,
      title: example.title,
      files: example.files.map((name) => `${example.folder}/${name}`),
    }));
  },

  createWorker() {
    const worker = new Worker("static/worker.js");
    worker.onmessage = (event) => this.handleMessage(event.data);
    worker.onerror = () => {
      const error = new Error("The browser worker failed unexpectedly.");
      error.code = "unexpected";
      this.failPending(error);
    };
    return worker;
  },

  failPending(error) {
    if (this.workerRejectReady) {
      const reject = this.workerRejectReady;
      this.workerResolveReady = null;
      this.workerRejectReady = null;
      reject(error);
    }
    if (this.pending) {
      const { reject } = this.pending;
      this.pending = null;
      reject(error);
    }
  },

  handleMessage(msg) {
    switch (msg.type) {
      case "ready":
        if (this.workerResolveReady) {
          this.workerResolveReady();
          this.workerResolveReady = null;
          this.workerRejectReady = null;
        }
        return;
      case "status":
        if (this.onProgress) this.onProgress(msg);
        return;
      case "result":
        if (this.pending && this.pending.id === msg.id) {
          const { resolve } = this.pending;
          this.pending = null;
          resolve({ metrics: msg.metrics, files: msg.files });
        }
        return;
      case "error": {
        const error = new Error(msg.message || "Stitching failed.");
        error.code = msg.code || "unexpected";
        error.details = msg.details || null;
        if (msg.id == null && this.workerRejectReady) {
          const reject = this.workerRejectReady;
          this.workerResolveReady = null;
          this.workerRejectReady = null;
          reject(error);
        } else if (this.pending && this.pending.id === msg.id) {
          const { reject } = this.pending;
          this.pending = null;
          reject(error);
        }
        return;
      }
      default:
        return;
    }
  },

  // Creates the worker on the very first call and reuses it afterwards, so a
  // second run skips straight to the "inputs" stage instead of reloading
  // Pyodide. A failed init discards the worker so the next run starts clean.
  async ensureWorker() {
    if (this.worker && this.workerReady) {
      await this.workerReady;
      return this.worker;
    }
    this.worker = this.createWorker();
    this.workerReady = new Promise((resolve, reject) => {
      this.workerResolveReady = resolve;
      this.workerRejectReady = reject;
    }).catch((error) => {
      this.worker = null;
      this.workerReady = null;
      throw error;
    });
    // The worker's own location (static/worker.js) is not the page's
    // location, so it needs the page's base URL to resolve
    // "panorama_stitching/*.py" and the demo-pair images the same way this
    // script does (both relative to the page, which also works one level
    // deep under a GitHub Pages project sub-path).
    const baseUrl = new URL(".", location.href).href;
    this.worker.postMessage({ type: "init", baseUrl });
    await this.workerReady;
    return this.worker;
  },

  // Terminates the worker outright (there is no cooperative-cancel message
  // in the protocol) and rejects whatever run was in flight; the next run
  // creates a fresh worker via ensureWorker().
  cancel() {
    if (this.worker) this.worker.terminate();
    this.worker = null;
    this.workerReady = null;
    this.failPending(localizedError("status.cancelled"));
    this.onProgress = null;
  },

  async stitch(input, onProgress) {
    let inputs;
    const transfer = [];

    if (input.mode === "example") {
      const example = STATIC_EXAMPLES.find((item) => item.id === input.exampleId);
      if (!example) throw localizedError("status.exampleMissing");
      inputs = example.files.map((name) => ({ url: `${example.folder}/${name}` }));
    } else {
      const buffers = await Promise.all(input.files.map((file) => file.arrayBuffer()));
      inputs = input.files.map((file, index) => ({ name: file.name, bytes: buffers[index] }));
      transfer.push(...buffers);
    }

    this.onProgress = onProgress;
    try {
      const worker = await this.ensureWorker();
      const id = `job-${++this.jobCounter}`;
      const data = await new Promise((resolve, reject) => {
        this.pending = { id, resolve, reject };
        worker.postMessage({ type: "stitch", id, inputs, maxSide: BROWSER_MAX_SIDE }, transfer);
      });

      const toUrl = (buffer) => URL.createObjectURL(new Blob([buffer], { type: "image/jpeg" }));
      const files = {
        panorama: toUrl(data.files.panorama),
        keypoints: data.files.keypoints.map(toUrl),
        matches: data.files.matches.map(toUrl),
        ransac: data.files.ransac.map(toUrl),
        pairs: data.files.pairs,
      };
      return { metrics: data.metrics, files };
    } finally {
      this.onProgress = null;
    }
  },
};

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
  const uploadReady = state.mode !== "upload" || (state.photos.length >= MIN_PHOTOS && state.photos.length <= MAX_PHOTOS);
  els.runButton.disabled = state.running || !uploadReady;
  els.progressBar.hidden = !state.running;
  els.cancelButton.hidden = !(state.running && state.backend && state.backend.name === "browser");
}

// The <progress> element is indeterminate whenever it has no value/max, which
// is what a fresh run should show until the first staged status arrives.
function resetProgress() {
  els.progressBar.removeAttribute("value");
  els.progressBar.removeAttribute("max");
}

function setProgress(step, total) {
  if (!step || !total) return;
  els.progressBar.max = total;
  els.progressBar.value = step;
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
  renderPhotoGrid();
  renderExampleOptions();
  renderDetailTabs();
  if (state.lastMetrics) setMetrics(state.lastMetrics);
  renderRunButton();
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
  renderRunButton();
}

function refreshExamplePreview() {
  const example = state.examples.find((item) => item.id === els.exampleSelect.value);
  if (!example) return;
  els.examplePreview.replaceChildren();
  example.files.forEach((url, index) => {
    const figure = document.createElement("figure");
    const image = document.createElement("img");
    image.src = url;
    image.alt = t("example.photoAlt", { index: index + 1 });
    const caption = document.createElement("figcaption");
    caption.textContent = t("example.photoCaption", { index: index + 1 });
    figure.append(image, caption);
    els.examplePreview.appendChild(figure);
  });
}

// ------------------------------------------------------------- photo grid --

function addPhotos(fileList) {
  const incoming = Array.from(fileList || []);
  if (incoming.length === 0) return;
  const imageFiles = incoming.filter((file) => file.type.startsWith("image/"));
  if (imageFiles.length === 0) {
    setStatus(() => t("upload.invalidFile"), true);
    return;
  }
  const room = Math.max(MAX_PHOTOS - state.photos.length, 0);
  const accepted = imageFiles.slice(0, room);
  if (imageFiles.length > accepted.length) {
    setStatus(() => t("error.too_many_images"), true);
  }
  accepted.forEach((file) => {
    state.photos.push({ file, previewUrl: URL.createObjectURL(file) });
  });
  renderPhotoGrid();
}

function removePhoto(index) {
  const [removed] = state.photos.splice(index, 1);
  if (removed) URL.revokeObjectURL(removed.previewUrl);
  renderPhotoGrid();
}

function renderPhotoGrid() {
  els.photoGrid.innerHTML = state.photos
    .map(
      (photo, index) => `
        <div class="photo-item">
          <span class="photo-index">${index + 1}</span>
          <img class="photo-thumb" src="${photo.previewUrl}" alt="" />
          <button type="button" class="photo-remove" data-index="${index}" aria-label="${t("upload.removeAriaLabel", { index: index + 1 })}">
            <span aria-hidden="true">&times;</span>
          </button>
        </div>`
    )
    .join("");
  // Browsers other than Safari cannot render HEIC previews; show the format instead.
  els.photoGrid.querySelectorAll(".photo-thumb").forEach((image, index) => {
    image.addEventListener("error", () => {
      const fallback = document.createElement("div");
      fallback.className = "photo-thumb photo-thumb-fallback";
      const name = state.photos[index]?.file.name || "";
      fallback.textContent = (name.split(".").pop() || "?").toUpperCase().slice(0, 5);
      image.replaceWith(fallback);
    });
  });
  els.photoCounter.textContent = t("upload.counter", { count: state.photos.length, max: MAX_PHOTOS });
  renderRunButton();
}

function wireDropZone(zoneEl) {
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
    addPhotos(event.dataTransfer && event.dataTransfer.files);
  });
}

// ------------------------------------------------------------- results ui --

function setMetrics(metrics) {
  const locale = state.lang === "tr" ? "tr-TR" : "en-US";
  els.mKeypoints.textContent = metrics.totalKeypoints.toLocaleString(locale);
  const pairsUsed = metrics.pairs ? metrics.pairs.length : 0;
  els.mPairs.textContent = t("metrics.pairsValue", { used: pairsUsed, evaluated: metrics.pairsEvaluated });
  els.mInliers.textContent = metrics.totalInliers.toLocaleString(locale);
  els.mOrder.textContent = (metrics.order || []).join(" → ");
}

// Builds the chip list from the current result files: a "Matches i+j" and
// "RANSAC i+j" chip per stitched pair (in files.pairs order), then a
// "Keypoints k" chip per input photo.
function buildDetailTabs(files) {
  const tabs = [];
  (files.pairs || []).forEach(([i, j], index) => {
    tabs.push({ target: `matches:${index}`, label: t("tabs.matchesPair", { i, j }) });
    tabs.push({ target: `ransac:${index}`, label: t("tabs.ransacPair", { i, j }) });
  });
  (files.keypoints || []).forEach((_, index) => {
    tabs.push({ target: `keypoints:${index}`, label: t("tabs.keypointsImage", { i: index + 1 }) });
  });
  return tabs;
}

function detailUrl(target) {
  if (!target) return null;
  const [kind, indexText] = target.split(":");
  const index = Number(indexText);
  const files = state.resultFiles;
  if (kind === "matches") return files.matches && files.matches[index];
  if (kind === "ransac") return files.ransac && files.ransac[index];
  if (kind === "keypoints") return files.keypoints && files.keypoints[index];
  return null;
}

// Rebuilds the chip strip, re-localizing every label; keeps whatever chip
// was already selected (used on a language switch) or defaults to the first
// one (used right after a run).
function renderDetailTabs() {
  const tabs = buildDetailTabs(state.resultFiles);
  const activeTarget = tabs.some((tab) => tab.target === state.activeDetailTarget) ? state.activeDetailTarget : tabs[0]?.target || null;
  els.detailTabs.innerHTML = tabs
    .map((tab) => {
      const active = tab.target === activeTarget;
      return `<button class="tab${active ? " active" : ""}" type="button" role="tab" aria-selected="${active}" data-target="${tab.target}">${tab.label}</button>`;
    })
    .join("");
  state.activeDetailTarget = activeTarget;
}

function selectDetail(target) {
  const src = detailUrl(target);
  if (!src) return;
  state.activeDetailTarget = target;
  els.detailTabs.querySelectorAll(".tab").forEach((tab) => {
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
    .flat()
    .filter((value) => typeof value === "string" && value.startsWith("blob:"))
    .forEach((url) => URL.revokeObjectURL(url));
  state.resultFiles = data.files;
  state.lastMetrics = data.metrics;
  state.activeDetailTarget = null; // force the chip strip back to the first one
  els.panoramaEmpty.hidden = true;
  els.panoramaImage.hidden = false;
  els.panoramaImage.src = withCacheBuster(data.files.panorama);
  els.downloadButton.href = data.files.panorama;
  els.downloadButton.classList.remove("disabled");
  setMetrics(data.metrics);
  renderDetailTabs();
  if (state.activeDetailTarget) selectDetail(state.activeDetailTarget);
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
    if (state.photos.length < MIN_PHOTOS || state.photos.length > MAX_PHOTOS) {
      setStatus(() => t("status.needPhotos"), true);
      return;
    }
  }

  state.running = true;
  resetProgress();
  renderRunButton();

  try {
    if (state.mode === "upload") {
      setStatus(() => t("status.preparingPhotos"));
      const canDefer = state.backend.name === "server";
      input.files = await Promise.all(
        state.photos.map((photo) => shrinkForUpload(photo.file, canDefer))
      );
    }
    if (state.backend.name === "server") {
      setStatus(() => t("status.computingServer"));
    }
    // Browser-mode progress arrives as { stage, step, total, barStep,
    // barTotal } from the worker (see static/worker.js); server mode never
    // calls this back.
    const data = await state.backend.stitch(input, (entry) => {
      setStatus(() => t(`stage.${entry.stage}`, { step: entry.step, total: entry.total }));
      setProgress(entry.barStep, entry.barTotal);
    });
    setResult(data);
    setStatus(() => buildDoneMessage(data.metrics, state.backend.name));
  } catch (error) {
    setStatus(() => resolveErrorMessage(error), true);
  } finally {
    state.running = false;
    resetProgress();
    renderRunButton();
  }
}

function cancelStitching() {
  if (!state.running || !state.backend || state.backend.name !== "browser") return;
  state.backend.cancel();
}

els.langButtons.forEach((button) => {
  button.addEventListener("click", () => setLang(button.dataset.lang));
});

els.modes.forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

els.exampleSelect.addEventListener("change", refreshExamplePreview);
els.runButton.addEventListener("click", runStitching);
els.cancelButton.addEventListener("click", cancelStitching);

els.detailTabs.addEventListener("click", (event) => {
  const tab = event.target.closest(".tab");
  if (tab) selectDetail(tab.dataset.target);
});

els.photosInput.addEventListener("change", () => {
  addPhotos(els.photosInput.files);
  els.photosInput.value = "";
});
els.photoGrid.addEventListener("click", (event) => {
  const button = event.target.closest(".photo-remove");
  if (button) removePhoto(Number(button.dataset.index));
});
wireDropZone(els.dropZone);

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
