// The page works against two back ends:
//  - the Python server in app.py (POST api/stitch), when it is running;
//  - the same Python modules executed inside the browser through Pyodide
//    (CPython + OpenCV compiled to WebAssembly) when there is no server,
//    which is how the GitHub Pages deployment runs.
// Both return the same shape: { metrics, files: { panorama, matches, ... } }.

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js";
const PY_MODULES = ["errors.py", "matcher.py", "homografi.py", "birlestirme.py", "panorama_pipeline.py"];
// Longest input side used in the browser; SIFT in WebAssembly is a few times
// slower than native, and this keeps a pair under ~6 s.
const BROWSER_MAX_SIDE = 1400;
// Mirror of panorama_pipeline.EXAMPLES for the server-less mode (a test keeps
// the two lists in sync).
const STATIC_EXAMPLES = [
  { id: "clock", title: "Saat Kulesi", folder: "images/Clock", left: "sol1.jpg", right: "sag1.jpg" },
  { id: "school", title: "Okul Bahcesi", folder: "images/SchoolImage", left: "sol2.jpg", right: "sag2.jpg" },
  { id: "street", title: "Test Goruntusu", folder: "images/test1", left: "s1.jpg", right: "s2.jpg" },
];

const state = {
  mode: "example",
  examples: [],
  resultFiles: {},
  backend: null,
};

const els = {
  modes: document.querySelectorAll(".mode"),
  examplePanel: document.querySelector("#examplePanel"),
  uploadPanel: document.querySelector("#uploadPanel"),
  exampleSelect: document.querySelector("#exampleSelect"),
  exampleLeft: document.querySelector("#exampleLeft"),
  exampleRight: document.querySelector("#exampleRight"),
  leftImage: document.querySelector("#leftImage"),
  rightImage: document.querySelector("#rightImage"),
  leftFileName: document.querySelector("#leftFileName"),
  rightFileName: document.querySelector("#rightFileName"),
  runButton: document.querySelector("#runButton"),
  status: document.querySelector("#status"),
  panoramaImage: document.querySelector("#panoramaImage"),
  detailImage: document.querySelector("#detailImage"),
  tabs: document.querySelectorAll(".tab"),
  downloadButton: document.querySelector("#downloadButton"),
  mLeft: document.querySelector("#mLeft"),
  mRight: document.querySelector("#mRight"),
  mMatches: document.querySelector("#mMatches"),
  mInliers: document.querySelector("#mInliers"),
};

function setStatus(message, isError = false) {
  els.status.textContent = message;
  els.status.classList.toggle("error", isError);
}

// Let the browser paint the status line before a long synchronous job.
function nextPaint() {
  return new Promise((resolve) => setTimeout(resolve, 30));
}

function withCacheBuster(url) {
  return url.startsWith("blob:") ? url : `${url}?t=${Date.now()}`;
}

// ---------------------------------------------------------------- server ---

const serverBackend = {
  name: "server",

  async listExamples() {
    const response = await fetch("api/examples");
    if (!response.ok) throw new Error("Örnekler alınamadı.");
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
    if (!data.ok) throw new Error(data.error || "İşlem tamamlanamadı.");
    return { metrics: data.metrics, files: data.files };
  },
};

// --------------------------------------------------------------- browser ---

function loadScript(url) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = url;
    script.onload = resolve;
    script.onerror = () => reject(new Error("Pyodide indirilemedi; bağlantıyı kontrol edin."));
    document.head.appendChild(script);
  });
}

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Dosya alınamadı: ${url}`);
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
        onStatus("Python ve OpenCV tarayıcıya yükleniyor (ilk açılışta yaklaşık 20 MB)...");
        await loadScript(PYODIDE_URL);
        const pyodide = await window.loadPyodide();
        await pyodide.loadPackage(["numpy", "opencv-python"]);
        pyodide.FS.mkdirTree("/app/out");
        pyodide.FS.mkdirTree("/app/in");
        for (const name of PY_MODULES) {
          pyodide.FS.writeFile(`/app/${name}`, await fetchBytes(name));
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
      if (!example) throw new Error("Secilen ornek veri bulunamadi.");
      left = `/app/${example.folder}/${example.left}`;
      right = `/app/${example.folder}/${example.right}`;
      onStatus("Örnek görseller alınıyor...");
      await this.fetchIntoFS(pyodide, `${example.folder}/${example.left}`, left);
      await this.fetchIntoFS(pyodide, `${example.folder}/${example.right}`, right);
    } else {
      pyodide.FS.mkdirTree(inDir);
      left = `${inDir}/sol${extensionOf(input.leftFile.name)}`;
      right = `${inDir}/sag${extensionOf(input.rightFile.name)}`;
      pyodide.FS.writeFile(left, new Uint8Array(await input.leftFile.arrayBuffer()));
      pyodide.FS.writeFile(right, new Uint8Array(await input.rightFile.arrayBuffer()));
    }

    onStatus("SIFT noktaları, eşleşmeler ve homografi tarayıcıda hesaplanıyor (birkaç saniye)...");
    await nextPaint();

    // Same call the server makes; PanoramaError becomes {ok: false, error}.
    const script = `
import json
from panorama_pipeline import PanoramaError, stitch_pair
try:
    sonuc = stitch_pair(${JSON.stringify(left)}, ${JSON.stringify(right)},
                        ${JSON.stringify(outDir)}, max_side=${BROWSER_MAX_SIDE})
    sonuc["ok"] = True
except PanoramaError as hata:
    sonuc = {"ok": False, "error": str(hata)}
json.dumps(sonuc)
`;
    const result = JSON.parse(pyodide.runPython(script));
    if (!result.ok) {
      this.removeTree(pyodide, outDir);
      this.removeTree(pyodide, inDir);
      throw new Error(result.error || "İşlem tamamlanamadı.");
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

function setMode(mode) {
  state.mode = mode;
  els.modes.forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  els.examplePanel.classList.toggle("hidden", mode !== "example");
  els.uploadPanel.classList.toggle("hidden", mode !== "upload");
}

function refreshExamplePreview() {
  const example = state.examples.find((item) => item.id === els.exampleSelect.value);
  if (!example) return;
  els.exampleLeft.src = example.left;
  els.exampleRight.src = example.right;
}

function setMetrics(metrics) {
  els.mLeft.textContent = metrics.leftKeypoints.toLocaleString("tr-TR");
  els.mRight.textContent = metrics.rightKeypoints.toLocaleString("tr-TR");
  els.mMatches.textContent = metrics.goodMatches.toLocaleString("tr-TR");
  els.mInliers.textContent = metrics.inliers.toLocaleString("tr-TR");
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
  els.panoramaImage.src = withCacheBuster(data.files.panorama);
  els.downloadButton.href = data.files.panorama;
  els.downloadButton.classList.remove("disabled");
  setMetrics(data.metrics);
  selectDetail("matches");
}

async function loadExamples() {
  state.examples = await state.backend.listExamples();
  els.exampleSelect.innerHTML = state.examples
    .map((example) => `<option value="${example.id}">${example.title}</option>`)
    .join("");
  refreshExamplePreview();
}

async function runStitching() {
  const input = { mode: state.mode };
  if (state.mode === "example") {
    input.exampleId = els.exampleSelect.value;
  } else {
    if (!els.leftImage.files[0] || !els.rightImage.files[0]) {
      setStatus("Sol ve sağ görseli seçmen gerekiyor.", true);
      return;
    }
    input.leftFile = els.leftImage.files[0];
    input.rightFile = els.rightImage.files[0];
  }

  els.runButton.disabled = true;
  setStatus("SIFT noktaları, eşleşmeler ve homografi hesaplanıyor...");

  try {
    const data = await state.backend.stitch(input, setStatus);
    setResult(data);
    let message = `Panorama hazır: ${data.metrics.panoramaWidth} x ${data.metrics.panoramaHeight}px`;
    if (data.metrics.inputScale && data.metrics.inputScale < 1) {
      const percent = Math.round(data.metrics.inputScale * 100);
      message += ` (girdiler %${percent} boyuta küçültüldü)`;
    }
    if (state.backend.name === "browser") message += " · tarayıcıda hesaplandı";
    setStatus(message);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    els.runButton.disabled = false;
  }
}

els.modes.forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

els.exampleSelect.addEventListener("change", refreshExamplePreview);
els.runButton.addEventListener("click", runStitching);
els.tabs.forEach((tab) => tab.addEventListener("click", () => selectDetail(tab.dataset.target)));

els.leftImage.addEventListener("change", () => {
  els.leftFileName.textContent = els.leftImage.files[0]?.name || "Dosya seç";
});

els.rightImage.addEventListener("change", () => {
  els.rightFileName.textContent = els.rightImage.files[0]?.name || "Dosya seç";
});

async function boot() {
  state.backend = await pickBackend();
  await loadExamples();
  if (state.backend.name === "server") {
    // With a server the run is cheap: refresh the previews and metrics on load.
    await runStitching();
  } else {
    setStatus("Sunucu yok: işlem tarayıcıda yapılır. Başlamak için Panoramayı Oluştur'a basın.");
  }
}

boot().catch((error) => setStatus(error.message, true));
