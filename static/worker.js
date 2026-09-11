// Runs the panorama_stitching pipeline off the main thread via Pyodide
// (CPython + OpenCV compiled to WebAssembly), so a multi-second SIFT/RANSAC/
// blend pass never freezes the page. This file owns Pyodide end to end:
// loading the runtime, loading numpy/opencv-python, fetching the pipeline's
// own .py files into the virtual file system, and running stitch_set().
//
// Message protocol (see static/app.js's browserBackend for the other side):
//   main -> worker
//     { type: "init", baseUrl }
//     { type: "stitch", id, inputs, maxSide }
//       inputs is a list of 2-6 entries, each either { url } (a demo pair
//       photo, fetched here relative to baseUrl) or { name, bytes } (an
//       uploaded file, bytes is a transferred ArrayBuffer). Order in the
//       list is arbitrary; the pipeline recovers the panorama arrangement.
//   worker -> main
//     { type: "ready" }
//     { type: "status", id, stage, step, total, barStep, barTotal }
//       stage/step/total label the current step within its own phase (the
//       4 worker setup stages runtime/packages/modules/inputs count 1-4 of
//       4; the 5 pipeline stages features/matching/homography/blending/
//       saving report the step/total the Python pipeline itself computed
//       for the whole photo set). barStep/barTotal are the same progress
//       expressed as one running total (4 setup steps plus the pipeline's
//       own total) for driving the overall progress bar.
//     { type: "result", id, metrics, files }
//       files is { panorama, keypoints: [...], matches: [...], ransac: [...],
//       pairs: [[i, j], ...] }, every image entry a transferred ArrayBuffer.
//     { type: "error", id, code, message, details }
//       id is null for a failure during "init" (no stitch job is in flight
//       yet); otherwise it matches the "stitch" message's id. details
//       mirrors PanoramaError.details ({"pair": [i, j]} or {"image": k}, or
//       null).

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js";
const PYODIDE_INDEX_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
const PY_PACKAGE_DIR = "panorama_stitching";
const PY_MODULES = ["__init__.py", "errors.py", "features.py", "matching.py", "homography.py", "blending.py", "projection.py", "pipeline.py"];

// The worker's own setup work (loading Pyodide, numpy/opencv-python, the
// pipeline's .py files, then writing the input photos into the virtual FS)
// always takes exactly these 4 steps, in this order; see app.js's stage.*
// i18n keys for the matching labels. The pipeline itself reports the
// remaining features/matching/homography/blending/saving steps through the
// report_progress callback below, with a step/total that spans the whole
// photo set (see panorama_stitching.pipeline.stitch_set).
const SETUP_STEPS = { runtime: 1, packages: 2, modules: 3, inputs: 4 };
const SETUP_TOTAL = 4;

let baseUrl = "";
let pyodide = null;
let pyodideReadyPromise = null;

// barTotalOverride lets the "inputs" stage (the first point at which the
// photo count, and so the pipeline's own total, is known) report a bar
// total that already includes the pipeline's steps, so the bar does not
// jump when pipeline reporting takes over. Earlier setup stages don't know
// the photo count yet and fall back to the plain 4-step setup total.
function reportSetupStage(id, stage, barTotalOverride) {
  const step = SETUP_STEPS[stage];
  postMessage({
    type: "status",
    id,
    stage,
    step,
    total: SETUP_TOTAL,
    barStep: step,
    barTotal: barTotalOverride ?? SETUP_TOTAL,
  });
}

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const error = new Error(`Could not fetch file: ${url}`);
    error.code = "fetch_failed";
    throw error;
  }
  return new Uint8Array(await response.arrayBuffer());
}

function extensionOf(name) {
  const match = /\.[a-z0-9]+$/i.exec(name || "");
  return match ? match[0].toLowerCase() : ".jpg";
}

function removeTree(instance, dir) {
  let entries;
  try {
    entries = instance.FS.readdir(dir).filter((name) => name !== "." && name !== "..");
  } catch (error) {
    return;
  }
  for (const name of entries) instance.FS.unlink(`${dir}/${name}`);
  instance.FS.rmdir(dir);
}

// Loads Pyodide, numpy/opencv-python and the pipeline's own .py files
// exactly once per worker instance; later "stitch" messages reuse the
// result, which is what lets a second run skip straight to the "inputs"
// stage.
async function ensurePyodide(id) {
  if (pyodide) return pyodide;
  if (!pyodideReadyPromise) {
    pyodideReadyPromise = (async () => {
      reportSetupStage(id, "runtime");
      importScripts(PYODIDE_URL);
      const instance = await loadPyodide({ indexURL: PYODIDE_INDEX_URL });

      reportSetupStage(id, "packages");
      await instance.loadPackage(["numpy", "opencv-python"]);

      reportSetupStage(id, "modules");
      instance.FS.mkdirTree(`/app/${PY_PACKAGE_DIR}`);
      instance.FS.mkdirTree("/app/in");
      instance.FS.mkdirTree("/app/out");
      for (const name of PY_MODULES) {
        const bytes = await fetchBytes(`${baseUrl}${PY_PACKAGE_DIR}/${name}`);
        instance.FS.writeFile(`/app/${PY_PACKAGE_DIR}/${name}`, bytes);
      }
      instance.runPython("import sys\nsys.path.insert(0, '/app')");

      pyodide = instance;
      return instance;
    })().catch((error) => {
      pyodideReadyPromise = null;
      if (!error.code) error.code = "pyodide_failed";
      throw error;
    });
  }
  return pyodideReadyPromise;
}

async function writeInput(instance, dir, index, spec) {
  const bytes = spec.url ? await fetchBytes(`${baseUrl}${spec.url}`) : new Uint8Array(spec.bytes);
  const ext = extensionOf(spec.url || spec.name);
  const path = `${dir}/photo_${index}${ext}`;
  instance.FS.writeFile(path, bytes);
  return path;
}

async function runStitch(msg) {
  const { id, inputs, maxSide } = msg;
  const instance = await ensurePyodide(id);

  // Known as soon as the "stitch" message arrives, independent of Pyodide
  // state; matches panorama_stitching.pipeline.stitch_set's own step count.
  const n = inputs.length;
  // Mirrors pipeline._set_progress_total: features per photo, every pair
  // matched, the tree edges re-matched on the cylinder for 3+ photos, then
  // homography, blending and saving. Only the first bar estimate uses this;
  // every status message carries the pipeline's own total afterwards.
  const pipelineTotal = n + (n * (n - 1)) / 2 + (n > 2 ? n - 1 : 0) + 3;
  reportSetupStage(id, "inputs", SETUP_TOTAL + pipelineTotal);

  const jobDir = String(id).replace(/[^a-zA-Z0-9_-]/g, "_");
  const inDir = `/app/in/${jobDir}`;
  const outDir = `/app/out/${jobDir}`;
  instance.FS.mkdirTree(inDir);
  instance.FS.mkdirTree(outDir);

  let paths;
  try {
    paths = [];
    for (let i = 0; i < inputs.length; i++) {
      paths.push(await writeInput(instance, inDir, i + 1, inputs[i]));
    }
  } catch (error) {
    removeTree(instance, inDir);
    removeTree(instance, outDir);
    throw error;
  }

  // Bridges the Python pipeline's own progress callback (stitch_set's
  // `progress` parameter) onto the same status message shape this file
  // reports its own setup stages with. step/total here are the pipeline's
  // own, spanning the whole photo set.
  instance.globals.set("report_progress", (stage, step, total) => {
    postMessage({
      type: "status",
      id,
      stage,
      step,
      total,
      barStep: SETUP_TOTAL + step,
      barTotal: SETUP_TOTAL + total,
    });
  });

  // Same call the server makes (see app.py); PanoramaError becomes
  // {ok: false, error, code, details}, and any other exception is reported
  // as "unexpected" rather than crashing the worker.
  const script = `
import json
from panorama_stitching.pipeline import PanoramaError, stitch_set
try:
    result = stitch_set(${JSON.stringify(paths)}, ${JSON.stringify(outDir)},
                        max_side=${JSON.stringify(maxSide)},
                        progress=report_progress)
    result["ok"] = True
except PanoramaError as exc:
    result = {"ok": False, "error": str(exc), "code": getattr(exc, "code", "unexpected"), "details": getattr(exc, "details", None)}
except Exception as exc:
    result = {"ok": False, "error": str(exc), "code": "unexpected", "details": None}
json.dumps(result)
`;
  let result;
  try {
    result = JSON.parse(instance.runPython(script));
  } finally {
    removeTree(instance, inDir);
  }

  if (!result.ok) {
    removeTree(instance, outDir);
    const error = new Error(result.error || "Stitching failed.");
    error.code = result.code || "unexpected";
    error.details = result.details || null;
    throw error;
  }

  const transfer = [];
  const readOne = (name) => {
    const bytes = instance.FS.readFile(`${outDir}/${name}`);
    // Copy out of the WASM heap into a standalone, transferable ArrayBuffer.
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    transfer.push(buffer);
    return buffer;
  };

  const files = {
    panorama: readOne(result.files.panorama),
    keypoints: result.files.keypoints.map(readOne),
    matches: result.files.matches.map(readOne),
    ransac: result.files.ransac.map(readOne),
    pairs: result.files.pairs,
  };
  removeTree(instance, outDir);

  return { metrics: result.metrics, files, transfer };
}

self.onmessage = async (event) => {
  const msg = event.data;

  if (msg.type === "init") {
    baseUrl = msg.baseUrl;
    try {
      await ensurePyodide(null);
      postMessage({ type: "ready" });
    } catch (error) {
      postMessage({ type: "error", id: null, code: error.code || "pyodide_failed", message: error.message || String(error), details: null });
    }
    return;
  }

  if (msg.type === "stitch") {
    try {
      const { metrics, files, transfer } = await runStitch(msg);
      postMessage({ type: "result", id: msg.id, metrics, files }, transfer);
    } catch (error) {
      postMessage({ type: "error", id: msg.id, code: error.code || "unexpected", message: error.message || String(error), details: error.details || null });
    }
    return;
  }
};
