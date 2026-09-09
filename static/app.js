const state = {
  mode: "example",
  examples: [],
  resultFiles: {},
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

function setMode(mode) {
  state.mode = mode;
  els.modes.forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
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
  els.tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.target === target));
  els.detailImage.src = `${src}?t=${Date.now()}`;
  els.detailImage.hidden = false;
}

function setResult(data) {
  state.resultFiles = data.files;
  els.panoramaImage.src = `${data.files.panorama}?t=${Date.now()}`;
  els.downloadButton.href = data.files.panorama;
  els.downloadButton.classList.remove("disabled");
  setMetrics(data.metrics);
  selectDetail("matches");
}

async function loadExamples() {
  const response = await fetch("/api/examples");
  const data = await response.json();
  state.examples = data.examples;
  els.exampleSelect.innerHTML = state.examples
    .map((example) => `<option value="${example.id}">${example.title}</option>`)
    .join("");
  refreshExamplePreview();
}

async function runStitching() {
  const formData = new FormData();
  formData.append("mode", state.mode);

  if (state.mode === "example") {
    formData.append("example", els.exampleSelect.value);
  } else {
    if (!els.leftImage.files[0] || !els.rightImage.files[0]) {
      setStatus("Sol ve sağ görseli seçmen gerekiyor.", true);
      return;
    }
    formData.append("leftImage", els.leftImage.files[0]);
    formData.append("rightImage", els.rightImage.files[0]);
  }

  els.runButton.disabled = true;
  setStatus("SIFT noktaları, eşleşmeler ve homografi hesaplanıyor...");

  try {
    const response = await fetch("/api/stitch", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!data.ok) {
      throw new Error(data.error || "İşlem tamamlanamadı.");
    }
    setResult(data);
    setStatus(`Panorama hazır: ${data.metrics.panoramaWidth} x ${data.metrics.panoramaHeight}px`);
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

// Örnekler yüklendikten sonra seçili veri setini otomatik çalıştır;
// böylece açılışta ara görseller ve metrikler taze üretilir.
loadExamples()
  .then(runStitching)
  .catch((error) => setStatus(error.message, true));
