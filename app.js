const NO_TEXT_NOTICE =
  "No product text was provided. SmartQC will only check image validity.";
const BACKEND_ERROR_MESSAGE =
  "Cannot connect to backend server. Please check whether FastAPI is running at http://localhost:8000.";
const STAGE2_SKIPPED_MESSAGE =
  "Stage 2 skipped because the image failed Stage 1 validation.";

const state = {
  file: null,
  previewUrl: null,
  loadedDraftImageUrl: null,
  localMetrics: null,
  lastDecision: null,
  lastResult: null,
  lastDraftId: null,
};

const elements = {
  imageInput: document.querySelector("#imageInput"),
  dropZone: document.querySelector("#dropZone"),
  previewFrame: document.querySelector("#previewFrame"),
  previewImage: document.querySelector("#previewImage"),
  imageMeta: document.querySelector("#imageMeta"),
  listingPreviewImage: document.querySelector("#listingPreviewImage"),
  listingPreviewPlaceholder: document.querySelector("#listingPreviewPlaceholder"),
  replaceImageButton: document.querySelector("#replaceImageButton"),
  removeImageButton: document.querySelector("#removeImageButton"),
  clearButton: document.querySelector("#clearButton"),
  saveDraftButton: document.querySelector("#saveDraftButton"),
  submitListingButton: document.querySelector("#submitListingButton"),
  draftList: document.querySelector("#draftList"),
  draftStatusText: document.querySelector("#draftStatusText"),
  refreshDraftsButton: document.querySelector("#refreshDraftsButton"),
  productNameInput: document.querySelector("#productNameInput"),
  categoryInput: document.querySelector("#categoryInput"),
  conditionInput: document.querySelector("#conditionInput"),
  priceInput: document.querySelector("#priceInput"),
  shortDescriptionInput: document.querySelector("#shortDescriptionInput"),
  descriptionInput: document.querySelector("#descriptionInput"),
  noTextWarning: document.querySelector("#noTextWarning"),
  wordCount: document.querySelector("#wordCount"),
  previewTitle: document.querySelector("#previewTitle"),
  previewPrice: document.querySelector("#previewPrice"),
  previewCategory: document.querySelector("#previewCategory"),
  previewDescription: document.querySelector("#previewDescription"),
  apiBaseInput: document.querySelector("#apiBaseInput"),
  analyzeButton: document.querySelector("#analyzeButton"),
  statusDot: document.querySelector("#statusDot"),
  statusText: document.querySelector("#statusText"),
  modelStatusText: document.querySelector("#modelStatusText"),
  modePill: document.querySelector("#modePill"),
  diagnosticsModePill: document.querySelector("#diagnosticsModePill"),
  finalDecision: document.querySelector("#finalDecision"),
  decisionCard: document.querySelector("#decisionCard"),
  decisionReason: document.querySelector("#decisionReason"),
  recommendationText: document.querySelector("#recommendationText"),
  summaryRecommendation: document.querySelector("#summaryRecommendation"),
  loadingIndicator: document.querySelector("#loadingIndicator"),
  errorMessage: document.querySelector("#errorMessage"),
  productValidityItem: document.querySelector("#productValidityItem"),
  productValidityIcon: document.querySelector("#productValidityIcon"),
  productValidityText: document.querySelector("#productValidityText"),
  textMatchItem: document.querySelector("#textMatchItem"),
  textMatchIcon: document.querySelector("#textMatchIcon"),
  textMatchText: document.querySelector("#textMatchText"),
  visualQualityItem: document.querySelector("#visualQualityItem"),
  visualQualityIcon: document.querySelector("#visualQualityIcon"),
  visualQualityText: document.querySelector("#visualQualityText"),
  stage1Status: document.querySelector("#stage1Status"),
  stage1Mode: document.querySelector("#stage1Mode"),
  matchScore: document.querySelector("#matchScore"),
  productScore: document.querySelector("#productScore"),
  invalidScore: document.querySelector("#invalidScore"),
  finalScore: document.querySelector("#finalScore"),
  stage1Reason: document.querySelector("#stage1Reason"),
  matchNote: document.querySelector("#matchNote"),
  stage2Status: document.querySelector("#stage2Status"),
  qualityScore: document.querySelector("#qualityScore"),
  qualityBar: document.querySelector("#qualityBar"),
  mosScore: document.querySelector("#mosScore"),
  stage2SkippedReason: document.querySelector("#stage2SkippedReason"),
  qualityNote: document.querySelector("#qualityNote"),
  summaryValidProduct: document.querySelector("#summaryValidProduct"),
  summaryTextMatched: document.querySelector("#summaryTextMatched"),
  summaryQualityAvailable: document.querySelector("#summaryQualityAvailable"),
  sharpnessBar: document.querySelector("#sharpnessBar"),
  exposureBar: document.querySelector("#exposureBar"),
  contrastBar: document.querySelector("#contrastBar"),
  sharpnessValue: document.querySelector("#sharpnessValue"),
  exposureValue: document.querySelector("#exposureValue"),
  contrastValue: document.querySelector("#contrastValue"),
  imageSignal: document.querySelector("#imageSignal"),
  descriptionSignal: document.querySelector("#descriptionSignal"),
  backendSignal: document.querySelector("#backendSignal"),
  mosModelState: document.querySelector("#mosModelState"),
  clipModelState: document.querySelector("#clipModelState"),
  lastResponseSignal: document.querySelector("#lastResponseSignal"),
  apiEndpointSignal: document.querySelector("#apiEndpointSignal"),
  shareUrlInput: document.querySelector("#shareUrlInput"),
  qrImage: document.querySelector("#qrImage"),
  qrFallback: document.querySelector("#qrFallback"),
  qrHelp: document.querySelector("#qrHelp"),
  refreshQrButton: document.querySelector("#refreshQrButton"),
  copyShareUrlButton: document.querySelector("#copyShareUrlButton"),
};

elements.apiBaseInput.value = window.IQS_API_BASE || inferBackendBase();

elements.imageInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) loadImageFile(file);
});

elements.dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.dropZone.classList.add("dragging");
});

elements.dropZone.addEventListener("dragleave", () => {
  elements.dropZone.classList.remove("dragging");
});

elements.dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.dropZone.classList.remove("dragging");
  const [file] = event.dataTransfer.files;
  if (file && file.type.startsWith("image/")) loadImageFile(file);
});

elements.replaceImageButton.addEventListener("click", () => elements.imageInput.click());
elements.removeImageButton.addEventListener("click", removeImage);
elements.clearButton.addEventListener("click", clearAll);
elements.saveDraftButton.addEventListener("click", saveDraft);
elements.submitListingButton.addEventListener("click", () => {
  showMessage("Listing can continue because SmartQC marked the image as ready to publish.", "notice");
});
elements.refreshDraftsButton?.addEventListener("click", loadDrafts);

document.querySelectorAll(".image-slot").forEach((slot) => {
  slot.addEventListener("click", () => elements.imageInput.click());
});

[
  elements.productNameInput,
  elements.categoryInput,
  elements.conditionInput,
  elements.priceInput,
  elements.shortDescriptionInput,
  elements.descriptionInput,
].forEach((input) => {
  input.addEventListener("input", () => {
    updateTextState();
    updateListingPreview();
    refreshReadyState();
  });
});

elements.analyzeButton.addEventListener("click", analyze);
elements.refreshQrButton.addEventListener("click", updateMobileAccess);
elements.copyShareUrlButton.addEventListener("click", copyShareUrl);
elements.qrImage.addEventListener("error", () => {
  elements.qrImage.hidden = true;
  elements.qrFallback.hidden = false;
  elements.qrHelp.textContent = "QR image service is unavailable. Copy the access URL instead.";
});

function loadImageFile(file) {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);

  resetResults();
  state.file = file;
  state.previewUrl = URL.createObjectURL(file);
  state.loadedDraftImageUrl = null;
  elements.previewImage.src = state.previewUrl;
  elements.listingPreviewImage.src = state.previewUrl;
  elements.previewFrame.hidden = false;
  elements.dropZone.hidden = true;
  elements.listingPreviewImage.hidden = false;
  elements.listingPreviewPlaceholder.hidden = true;
  elements.analyzeButton.disabled = false;
  elements.imageMeta.textContent = `${file.name} | ${formatBytes(file.size)}`;

  elements.previewImage.onload = async () => {
    elements.imageMeta.textContent = `${file.name} | ${formatBytes(file.size)} | ${elements.previewImage.naturalWidth}x${elements.previewImage.naturalHeight}`;
    state.localMetrics = await calculateLocalMetrics(elements.previewImage);
    updateLocalDiagnostics(state.localMetrics);
    setPrecheckState();
    updateListingPreview();
  };
}

function removeImage() {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = null;
  state.previewUrl = null;
  state.loadedDraftImageUrl = null;
  state.localMetrics = null;
  elements.imageInput.value = "";
  elements.previewImage.removeAttribute("src");
  elements.listingPreviewImage.removeAttribute("src");
  elements.previewFrame.hidden = true;
  elements.dropZone.hidden = false;
  elements.listingPreviewImage.hidden = true;
  elements.listingPreviewPlaceholder.hidden = false;
  elements.analyzeButton.disabled = true;
  resetResults();
  updateListingPreview();
}

function clearAll() {
  removeImage();
  state.lastDraftId = null;
  elements.productNameInput.value = "";
  elements.categoryInput.value = "";
  elements.conditionInput.value = "New";
  elements.priceInput.value = "";
  elements.shortDescriptionInput.value = "";
  elements.descriptionInput.value = "";
  updateTextState();
  updateListingPreview();
  resetResults();
}

async function analyze() {
  if (!state.file) {
    showMessage("Please upload a main product image first.");
    return;
  }

  const apiBase = getApiBase();
  setBusy(true, "Checking listing...");
  showMessage("");
  setAssistantProgress("Running Stage 1 CLIP gatekeeper...");

  if (!hasProductText()) showMessage(NO_TEXT_NOTICE, "notice");

  try {
    const result = await callBackend(apiBase);
    setBusy(true, "Generating final decision...");

    if (result?.success === false) {
      showModelError(result, apiBase);
      return;
    }

    applyBackendResult(result);
    setBackendState("connected", "Model API ready", "Connected");
  } catch (error) {
    showRequestError(error, apiBase);
  } finally {
    setBusy(false);
  }
}

async function callBackend(apiBase) {
  const formData = new FormData();
  formData.append("image", state.file);
  formData.append("product_name", elements.productNameInput.value.trim());
  formData.append("description", elements.descriptionInput.value.trim());

  const response = await fetch(`${apiBase}/analyze`, {
    method: "POST",
    body: formData,
  });

  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail = typeof payload === "string" ? payload : payload?.detail || JSON.stringify(payload);
    throw new Error(`Backend returned ${response.status}: ${detail}`);
  }

  return payload;
}

async function saveDraft() {
  const hasDraftContent =
    state.file ||
    state.loadedDraftImageUrl ||
    hasProductText() ||
    elements.categoryInput.value.trim() ||
    elements.priceInput.value.trim();

  if (!hasDraftContent) {
    showMessage("Add a product title or upload a main product image before saving a draft.");
    return;
  }

  const apiBase = getApiBase();
  setDraftBusy(true);

  try {
    const formData = new FormData();
    if (state.file) formData.append("image", state.file);
    formData.append("product_name", elements.productNameInput.value.trim());
    formData.append("category", elements.categoryInput.value.trim());
    formData.append("condition", elements.conditionInput.value.trim());
    formData.append("price", elements.priceInput.value.trim());
    formData.append("short_description", elements.shortDescriptionInput.value.trim());
    formData.append("description", elements.descriptionInput.value.trim());
    formData.append("decision", state.lastDecision || "");
    formData.append("stage1_json", JSON.stringify(state.lastResult?.stage1 || null));
    formData.append("stage2_json", JSON.stringify(state.lastResult?.stage2 || null));
    formData.append("summary_json", JSON.stringify(state.lastResult?.summary || null));
    formData.append("local_metrics_json", JSON.stringify(state.localMetrics || null));

    const response = await fetch(`${apiBase}/drafts`, {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok || payload?.success === false) {
      throw new Error(payload?.detail || payload?.error || `Backend returned ${response.status}`);
    }

    state.lastDraftId = payload?.draft?.id || null;
    showMessage(`Draft saved to backend${state.lastDraftId ? `: ${state.lastDraftId}` : ""}.`, "notice");
    await loadDrafts();
  } catch (error) {
    showMessage(`Draft save failed. ${error.message || BACKEND_ERROR_MESSAGE}`);
  } finally {
    setDraftBusy(false);
  }
}

async function loadDrafts() {
  if (!elements.draftList) return;

  const apiBase = getApiBase();
  elements.draftStatusText.textContent = "Loading saved drafts...";

  try {
    const response = await fetch(`${apiBase}/drafts?limit=12`);
    const payload = await response.json();
    if (!response.ok || payload?.success === false) {
      throw new Error(payload?.detail || payload?.error || `Backend returned ${response.status}`);
    }
    renderDrafts(payload.drafts || []);
    elements.draftStatusText.textContent = payload.drafts?.length
      ? "Saved drafts are stored by the backend."
      : "No saved drafts yet.";
  } catch (error) {
    elements.draftList.replaceChildren();
    elements.draftStatusText.textContent = "Saved drafts unavailable. Check whether the backend is running.";
    console.error(error);
  }
}

function renderDrafts(drafts) {
  elements.draftList.replaceChildren();

  if (!drafts.length) {
    const empty = document.createElement("p");
    empty.className = "draft-empty";
    empty.textContent = "No saved drafts yet.";
    elements.draftList.append(empty);
    return;
  }

  drafts.forEach((draft) => {
    const item = document.createElement("article");
    item.className = "draft-list-item";

    const copy = document.createElement("div");
    copy.className = "draft-copy";

    const title = document.createElement("strong");
    title.textContent = draft.product_name || "Untitled listing";

    const meta = document.createElement("span");
    meta.textContent = `${decisionLabel(normalizeDecision(draft.decision))} | ${formatDraftDate(draft.updated_at)}`;

    const details = document.createElement("small");
    details.textContent = [
      draft.category || "No category",
      draft.price ? formatPreviewPrice(draft.price) : "No price",
      draft.has_image ? "Image saved" : "No image",
    ].join(" | ");

    copy.append(title, meta, details);

    const openButton = document.createElement("button");
    openButton.className = "secondary-button small-button";
    openButton.type = "button";
    openButton.textContent = "Open";
    openButton.addEventListener("click", () => openDraft(draft.id));

    item.append(copy, openButton);
    elements.draftList.append(item);
  });
}

async function openDraft(draftId) {
  const apiBase = getApiBase();
  showMessage("Loading saved draft...", "notice");

  try {
    const response = await fetch(`${apiBase}/drafts/${encodeURIComponent(draftId)}`);
    const payload = await response.json();
    if (!response.ok || payload?.success === false) {
      throw new Error(payload?.detail || payload?.error || `Backend returned ${response.status}`);
    }

    const draft = payload.draft;
    state.lastDraftId = draft.id;
    elements.productNameInput.value = draft.product_name || "";
    elements.categoryInput.value = draft.category || "";
    elements.conditionInput.value = draft.condition || "New";
    elements.priceInput.value = draft.price || "";
    elements.shortDescriptionInput.value = draft.short_description || "";
    elements.descriptionInput.value = draft.description || "";

    if (draft.image_url) {
      await loadDraftImage(draft, apiBase);
    } else {
      removeImage();
    }

    updateTextState();
    updateListingPreview();

    if (draft.stage1 || draft.stage2 || draft.summary) {
      applyBackendResult({
        success: true,
        decision: draft.decision || "review",
        decision_reason: draft.summary?.recommendation || "Loaded from a saved backend draft.",
        stage1: draft.stage1 || {},
        stage2: draft.stage2 || {},
        summary: draft.summary || {},
      });
    } else {
      resetResults();
      if (state.file) setPrecheckState();
    }

    showMessage("Draft loaded from backend.", "notice");
  } catch (error) {
    showMessage(`Draft load failed. ${error.message || BACKEND_ERROR_MESSAGE}`);
  }
}

async function loadDraftImage(draft, apiBase) {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);

  const imageUrl = `${apiBase}${draft.image_url}`;
  state.file = null;
  state.previewUrl = null;
  state.loadedDraftImageUrl = imageUrl;
  state.localMetrics = null;
  elements.imageInput.value = "";
  elements.previewFrame.hidden = false;
  elements.dropZone.hidden = true;
  elements.listingPreviewImage.hidden = false;
  elements.listingPreviewPlaceholder.hidden = true;
  elements.imageMeta.textContent = draft.image_filename || "Saved draft image";

  elements.previewImage.onload = async () => {
    elements.imageMeta.textContent = `${draft.image_filename || "Saved draft image"} | ${
      elements.previewImage.naturalWidth
    }x${elements.previewImage.naturalHeight}`;
    state.localMetrics = await calculateLocalMetrics(elements.previewImage);
    updateLocalDiagnostics(state.localMetrics);
    updateListingPreview();
  };

  elements.previewImage.src = imageUrl;
  elements.listingPreviewImage.src = imageUrl;

  try {
    const response = await fetch(imageUrl);
    if (!response.ok) throw new Error(`Image request returned ${response.status}`);
    const blob = await response.blob();
    state.file = new File(
      [blob],
      draft.image_filename || "draft-product-image.jpg",
      { type: blob.type || "image/jpeg" },
    );
    elements.analyzeButton.disabled = false;
  } catch (error) {
    elements.analyzeButton.disabled = true;
    elements.imageMeta.textContent += " | Re-upload image to run SmartQC again";
    console.error(error);
  }
}

function applyBackendResult(result) {
  const stage1 = result?.stage1 || {};
  const stage2 = result?.stage2 || {};
  const summary = result?.summary || {};
  const decision = normalizeDecision(result?.decision);

  state.lastDecision = decision;
  state.lastResult = result;
  setDecision(decision, result?.decision_reason || summary.recommendation);
  applyStage1(stage1, summary);
  applyStage2(stage2);
  applySummary(summary);
  updateSubmitState(decision);

  elements.clipModelState.textContent = `Ran (${formatTechnicalStatus(stage1.status)})`;
  elements.lastResponseSignal.textContent = new Date().toLocaleTimeString();
  elements.apiEndpointSignal.textContent = getApiBase();

  if (!hasProductText()) showMessage(NO_TEXT_NOTICE, "notice");
}

function applyStage1(stage1, summary) {
  const status = stage1.status;
  setStage1Status(status);
  setStage1Checklist(status, summary);
  elements.stage1Mode.textContent = displayValue(stage1.mode);
  elements.matchScore.textContent = formatRawPercent(stage1.match_score);
  elements.productScore.textContent = formatRawPercent(stage1.product_score);
  elements.invalidScore.textContent = formatRawPercent(stage1.invalid_score);
  elements.finalScore.textContent = formatRawPercent(stage1.final_score);
  elements.stage1Reason.textContent = displayValue(stage1.reason);
  elements.matchNote.textContent = displayValue(stage1.message);
}

function applyStage2(stage2) {
  if (stage2.skipped) {
    setStage2Status("skipped");
    elements.qualityScore.textContent = "N/A";
    elements.qualityBar.style.width = "0%";
    elements.mosScore.textContent = "N/A";
    elements.stage2SkippedReason.textContent = displayValue(
      stage2.skipped_reason || stage2.message,
      STAGE2_SKIPPED_MESSAGE,
    );
    elements.qualityNote.textContent = displayValue(stage2.message, STAGE2_SKIPPED_MESSAGE);
    elements.mosModelState.textContent = "Skipped by Stage 1";
    setChecklistItem(
      elements.visualQualityItem,
      elements.visualQualityIcon,
      elements.visualQualityText,
      "warn",
      "Skipped",
      STAGE2_SKIPPED_MESSAGE,
    );
    return;
  }

  const quality = Number(stage2.quality_score);
  const qualityLabel = getQualityLabel(quality);
  const qualityState = Number.isFinite(quality)
    ? quality >= 70
      ? "ok"
      : quality >= 50
        ? "warn"
        : "fail"
    : "warn";
  setStage2Status("completed");
  elements.qualityScore.textContent = formatQualityScore(stage2.quality_score);
  elements.qualityBar.style.width = `${scoreToPercent(stage2.quality_score)}%`;
  elements.mosScore.textContent = formatMosScore(stage2.mos_score, stage2.mos_max_score);
  elements.stage2SkippedReason.textContent = "N/A";
  elements.qualityNote.textContent = displayValue(stage2.message);
  elements.mosModelState.textContent = `Ran (${formatMosScore(stage2.mos_score, stage2.mos_max_score)})`;
  setChecklistItem(
    elements.visualQualityItem,
    elements.visualQualityIcon,
    elements.visualQualityText,
    qualityState,
    qualityLabel,
    formatQualityScore(stage2.quality_score),
  );
}

function applySummary(summary) {
  const recommendation = displayValue(
    summary.recommendation,
    "Please review the image and product details before publishing.",
  );
  elements.recommendationText.textContent = mapRecommendation(recommendation, state.lastDecision);
  elements.summaryRecommendation.textContent = recommendation;
  elements.summaryValidProduct.textContent = formatBoolean(summary.is_valid_product_image);
  elements.summaryTextMatched.textContent = formatBoolean(summary.is_text_matched);
  elements.summaryQualityAvailable.textContent = formatBoolean(summary.quality_available);
}

function showRequestError(error, apiBase) {
  setBackendState("error", "Model API offline", "Unavailable");
  elements.backendSignal.textContent = "Unavailable";
  elements.mosModelState.textContent = "Request failed";
  elements.clipModelState.textContent = "Request failed";
  elements.lastResponseSignal.textContent = "No model response";
  elements.apiEndpointSignal.textContent = apiBase;
  setDecision("reject", "The backend did not return a usable model response.");
  setStage1Status("request_failed");
  setStage2Status("not_run");
  elements.qualityNote.textContent = BACKEND_ERROR_MESSAGE;
  elements.matchNote.textContent = BACKEND_ERROR_MESSAGE;
  setChecklistItem(
    elements.productValidityItem,
    elements.productValidityIcon,
    elements.productValidityText,
    "fail",
    "Request failed",
    "Backend unavailable.",
  );
  setChecklistItem(
    elements.textMatchItem,
    elements.textMatchIcon,
    elements.textMatchText,
    "fail",
    "Request failed",
    "Backend unavailable.",
  );
  setChecklistItem(
    elements.visualQualityItem,
    elements.visualQualityIcon,
    elements.visualQualityText,
    "fail",
    "Request failed",
    "Backend unavailable.",
  );
  updateSubmitState("reject");
  showMessage(BACKEND_ERROR_MESSAGE);
  console.error(error);
}

function showModelError(result, apiBase) {
  const stage = displayValue(result?.stage, "api");
  const errorText = displayValue(result?.error, "Model request failed.");
  const userMessage =
    stage === "stage1"
      ? "Backend is connected, but Stage 1 CLIP model weights are not available or failed to load."
      : stage === "stage2"
        ? "Backend is connected, but Stage 2 MOS model failed during inference."
        : "Backend is connected, but the API returned a model error.";

  setBackendState("connected", "Model API connected", "Model error");
  elements.backendSignal.textContent = "Connected";
  elements.apiEndpointSignal.textContent = apiBase;
  elements.lastResponseSignal.textContent = new Date().toLocaleTimeString();
  elements.clipModelState.textContent = stage === "stage1" ? "Load failed" : "Not completed";
  elements.mosModelState.textContent = stage === "stage2" ? "Load failed" : "Not run";
  setDecision("review", userMessage);
  setStage1Status(stage === "stage1" ? "request_failed" : "not_run");
  setStage2Status("not_run");
  elements.stage1Reason.textContent = userMessage;
  elements.matchNote.textContent = errorText;
  elements.qualityNote.textContent = stage === "stage1"
    ? "Stage 2 was not run because Stage 1 failed to load."
    : errorText;
  setChecklistItem(
    elements.productValidityItem,
    elements.productValidityIcon,
    elements.productValidityText,
    "warn",
    "Review",
    userMessage,
  );
  setChecklistItem(
    elements.textMatchItem,
    elements.textMatchIcon,
    elements.textMatchText,
    "warn",
    "Review",
    stage === "stage1" ? "Text-image consistency could not be checked." : "Waiting for model output.",
  );
  setChecklistItem(
    elements.visualQualityItem,
    elements.visualQualityIcon,
    elements.visualQualityText,
    "warn",
    "Skipped",
    "Quality scoring was not reached.",
  );
  updateSubmitState("review");
  showMessage(userMessage, "notice");
}

async function calculateLocalMetrics(image) {
  const maxWidth = 256;
  const scale = Math.min(1, maxWidth / image.naturalWidth);
  const width = Math.max(1, Math.round(image.naturalWidth * scale));
  const height = Math.max(1, Math.round(image.naturalHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0, width, height);

  const data = context.getImageData(0, 0, width, height).data;
  const luminance = new Float32Array(width * height);
  let sum = 0;
  let saturationSum = 0;

  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    const r = data[i] / 255;
    const g = data[i + 1] / 255;
    const b = data[i + 2] / 255;
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    luminance[p] = luma;
    sum += luma;
    saturationSum += Math.max(r, g, b) - Math.min(r, g, b);
  }

  const mean = sum / luminance.length;
  let variance = 0;
  for (const value of luminance) variance += (value - mean) ** 2;

  const contrastRaw = Math.sqrt(variance / luminance.length);
  let edgeSum = 0;
  let edgeCount = 0;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const idx = y * width + x;
      const gx = luminance[idx + 1] - luminance[idx - 1];
      const gy = luminance[idx + width] - luminance[idx - width];
      edgeSum += Math.sqrt(gx * gx + gy * gy);
      edgeCount += 1;
    }
  }

  const sharpness = clampPercent((edgeSum / Math.max(edgeCount, 1)) * 520);
  const exposure = clampPercent((1 - Math.abs(mean - 0.52) / 0.52) * 100);
  const contrast = clampPercent(contrastRaw * 360);
  const saturation = clampPercent((saturationSum / luminance.length) * 180);
  const qualityPercent = clampPercent(
    sharpness * 0.35 + exposure * 0.24 + contrast * 0.28 + saturation * 0.13,
  );

  return { qualityPercent, sharpness, exposure, contrast, saturation };
}

function updateLocalDiagnostics(metrics) {
  if (!metrics) return;
  elements.diagnosticsModePill.textContent = "Browser heuristic";
  setMetric(elements.sharpnessBar, elements.sharpnessValue, metrics.sharpness);
  setMetric(elements.exposureBar, elements.exposureValue, metrics.exposure);
  setMetric(elements.contrastBar, elements.contrastValue, metrics.contrast);
  elements.imageSignal.textContent = `${Math.round(
    (metrics.sharpness + metrics.exposure + metrics.contrast) / 3,
  )}%`;
}

function refreshReadyState() {
  if (!state.file) return;
  setPrecheckState();
}

function setPrecheckState() {
  if (!state.file) return;
  setStage1Status(hasProductText() ? "ready_text_image" : "ready_image_only");
  elements.stage1Mode.textContent = hasProductText() ? "text_image" : "image_only";
  elements.stage2Status.textContent = "Waiting for Stage 1";
  elements.recommendationText.textContent = "Run SmartQC to check this listing image.";
  elements.summaryRecommendation.textContent = "Run the listing check to generate a recommendation.";
}

function updateTextState() {
  updateWordCount();
  updateTextCompletenessSignal();
  elements.noTextWarning.hidden = hasProductText();
}

function updateListingPreview() {
  const title = elements.productNameInput.value.trim();
  const category = elements.categoryInput.value.trim();
  const price = elements.priceInput.value.trim();
  const shortDescription = elements.shortDescriptionInput.value.trim();
  const fullDescription = elements.descriptionInput.value.trim();
  const description = shortDescription || fullDescription;

  elements.previewTitle.textContent = title || "Product title will appear here";
  elements.previewCategory.textContent = category || "Category";
  elements.previewPrice.textContent = formatPreviewPrice(price);
  elements.previewDescription.textContent =
    description ? truncateText(description, 150) : "Description preview will appear here.";

  if (state.previewUrl || state.loadedDraftImageUrl) {
    elements.listingPreviewImage.hidden = false;
    elements.listingPreviewPlaceholder.hidden = true;
  } else {
    elements.listingPreviewImage.hidden = true;
    elements.listingPreviewPlaceholder.hidden = false;
  }
}

function resetResults() {
  state.lastDecision = null;
  state.lastResult = null;
  elements.modePill.textContent = "Waiting";
  elements.modePill.className = "mode-pill";
  elements.finalDecision.textContent = "Not checked";
  elements.decisionReason.textContent =
    "Upload a main image and run SmartQC to get a publishing recommendation.";
  elements.decisionCard.className = "decision-panel decision-waiting";
  setStage1Status("not_run");
  elements.stage1Mode.textContent = "N/A";
  elements.matchScore.textContent = "N/A";
  elements.productScore.textContent = "N/A";
  elements.invalidScore.textContent = "N/A";
  elements.finalScore.textContent = "N/A";
  elements.stage1Reason.textContent = "Stage 1 runs before image quality scoring.";
  elements.matchNote.textContent = "N/A";
  setStage2Status("not_run");
  elements.qualityScore.textContent = "N/A";
  elements.qualityBar.style.width = "0%";
  elements.mosScore.textContent = "N/A";
  elements.stage2SkippedReason.textContent = "N/A";
  elements.qualityNote.textContent = "Stage 2 is MOS overall image quality scoring.";
  elements.recommendationText.textContent = "Run SmartQC to receive listing guidance.";
  elements.summaryRecommendation.textContent = "Run the listing check to generate a recommendation.";
  elements.summaryValidProduct.textContent = "N/A";
  elements.summaryTextMatched.textContent = "N/A";
  elements.summaryQualityAvailable.textContent = "N/A";
  setMetric(elements.sharpnessBar, elements.sharpnessValue, 0, true);
  setMetric(elements.exposureBar, elements.exposureValue, 0, true);
  setMetric(elements.contrastBar, elements.contrastValue, 0, true);
  elements.diagnosticsModePill.textContent = "Not model";
  elements.imageSignal.textContent = "N/A";
  elements.descriptionSignal.textContent = "N/A";
  elements.backendSignal.textContent = getApiBase() ? "Ready" : "Not connected";
  elements.mosModelState.textContent = "Not run";
  elements.clipModelState.textContent = "Not run";
  elements.lastResponseSignal.textContent = "N/A";
  elements.apiEndpointSignal.textContent = getApiBase() || "N/A";
  setChecklistItem(
    elements.productValidityItem,
    elements.productValidityIcon,
    elements.productValidityText,
    "idle",
    "-",
    "Waiting for Stage 1.",
  );
  setChecklistItem(
    elements.textMatchItem,
    elements.textMatchIcon,
    elements.textMatchText,
    "idle",
    "-",
    "Waiting for product text and image.",
  );
  setChecklistItem(
    elements.visualQualityItem,
    elements.visualQualityIcon,
    elements.visualQualityText,
    "idle",
    "-",
    "Stage 2 runs only after Stage 1 allows it.",
  );
  updateSubmitState(null);
  showMessage("");
}

function setDecision(decision, reason) {
  const normalized = normalizeDecision(decision);
  elements.finalDecision.textContent = decisionLabel(normalized);
  elements.decisionReason.textContent = displayValue(reason, "Waiting for SmartQC output.");
  elements.decisionCard.className = `decision-panel decision-${normalized}`;
  elements.modePill.textContent = decisionLabel(normalized);
  elements.modePill.className = `mode-pill decision-${normalized}`;
}

function setStage1Status(status) {
  elements.stage1Status.textContent = stage1UserLabel(status);
  elements.stage1Status.className = `status-badge ${stage1StatusClass(status)}`;
}

function setStage2Status(status) {
  elements.stage2Status.textContent = stage2UserLabel(status);
  elements.stage2Status.className = `status-badge ${stage2StatusClass(status)}`;
}

function setStage1Checklist(status, summary) {
  if (status === "consistent") {
    setChecklistItem(
      elements.productValidityItem,
      elements.productValidityIcon,
      elements.productValidityText,
      "ok",
      "OK",
      "Image looks like a valid product photo.",
    );
    setChecklistItem(
      elements.textMatchItem,
      elements.textMatchIcon,
      elements.textMatchText,
      "ok",
      "OK",
      "Product image matches the listing text.",
    );
    return;
  }

  if (status === "missing_text_review") {
    setChecklistItem(
      elements.productValidityItem,
      elements.productValidityIcon,
      elements.productValidityText,
      summary?.is_valid_product_image === false ? "fail" : "warn",
      summary?.is_valid_product_image === false ? "Fail" : "Review",
      summary?.is_valid_product_image === false
        ? "Image may not be a valid product photo."
        : "Image-only validity check completed.",
    );
    setChecklistItem(
      elements.textMatchItem,
      elements.textMatchIcon,
      elements.textMatchText,
      "warn",
      "Review",
      "Product text is missing, so image-text consistency cannot be checked.",
    );
    return;
  }

  if (status === "review") {
    setChecklistItem(
      elements.productValidityItem,
      elements.productValidityIcon,
      elements.productValidityText,
      "warn",
      "Review",
      "Product image validity is borderline.",
    );
    setChecklistItem(
      elements.textMatchItem,
      elements.textMatchIcon,
      elements.textMatchText,
      "warn",
      "Review",
      "Product match needs manual review.",
    );
    return;
  }

  if (status === "invalid_image" || status === "inconsistent") {
    setChecklistItem(
      elements.productValidityItem,
      elements.productValidityIcon,
      elements.productValidityText,
      "fail",
      "Fail",
      status === "invalid_image"
        ? "Image may not be a valid product photo."
        : "Image evidence is too weak for the listing.",
    );
    setChecklistItem(
      elements.textMatchItem,
      elements.textMatchIcon,
      elements.textMatchText,
      "fail",
      "Fail",
      status === "inconsistent"
        ? "Image may not match the listing."
        : "Invalid image evidence blocks text-image approval.",
    );
  }
}

function setChecklistItem(item, icon, text, stateName, label, message) {
  item.className = `check-item check-${stateName}`;
  icon.textContent = label;
  text.textContent = message;
}

function setAssistantProgress(message) {
  elements.stage1Reason.textContent = message;
  setChecklistItem(
    elements.productValidityItem,
    elements.productValidityIcon,
    elements.productValidityText,
    "warn",
    "...",
    "Running product validity check.",
  );
  setChecklistItem(
    elements.textMatchItem,
    elements.textMatchIcon,
    elements.textMatchText,
    "warn",
    "...",
    hasProductText()
      ? "Checking image against product text."
      : "No product text, running image-only validity check.",
  );
  setChecklistItem(
    elements.visualQualityItem,
    elements.visualQualityIcon,
    elements.visualQualityText,
    "idle",
    "-",
    "Waiting for Stage 1 result.",
  );
}

function updateSubmitState(decision) {
  const canSubmit = decision === "pass";
  elements.submitListingButton.disabled = !canSubmit;
  elements.submitListingButton.classList.toggle("enabled", canSubmit);
}

function setMetric(bar, valueElement, value, empty = false) {
  const percent = clampPercent(value);
  bar.style.width = `${percent}%`;
  valueElement.textContent = empty ? "--" : `${Math.round(percent)}%`;
}

function updateWordCount() {
  elements.wordCount.textContent = String(countWords(elements.descriptionInput.value));
}

function updateTextCompletenessSignal() {
  const words = countWords(`${elements.productNameInput.value} ${elements.descriptionInput.value}`);
  elements.descriptionSignal.textContent = words
    ? `${Math.round(clampPercent(Math.min(words, 32) * 3.1))}%`
    : "N/A";
}

async function updateConnectionStatus() {
  const apiBase = getApiBase();
  if (!apiBase) {
    setBackendState("error", "No model API", "Not configured");
    elements.backendSignal.textContent = "Not connected";
    return;
  }

  setBackendState("connected", "Checking model API", "Checking...");
  elements.backendSignal.textContent = "Checking";

  try {
    const response = await fetch(`${apiBase}/health`, { method: "GET" });
    if (!response.ok) throw new Error(`Health check returned ${response.status}`);
    const health = await response.json();
    const stage1Available = health?.stage1_available !== false;
    const stage2Available = health?.stage2_available !== false;
    const clipLoaded = Boolean(health?.clip_loaded);
    const mosLoaded = Boolean(health?.mos_loaded);

    if (stage1Available && stage2Available) {
      setBackendState("connected", "Model API ready", clipLoaded || mosLoaded ? "Model loaded" : "Ready, lazy load");
      elements.backendSignal.textContent = "Ready";
    } else {
      setBackendState("error", "Model API incomplete", "Check backend");
      elements.backendSignal.textContent = "Check backend";
    }

    elements.mosModelState.textContent = stage2Available
      ? (mosLoaded ? "Loaded, not run" : "Ready, lazy load")
      : "Model file missing";
    elements.clipModelState.textContent = stage1Available
      ? (clipLoaded ? "Loaded, not run" : "Ready, lazy load")
      : "Unavailable";
    elements.apiEndpointSignal.textContent = apiBase;
  } catch (error) {
    setBackendState("error", "Model API offline", "Offline");
    elements.backendSignal.textContent = "Offline";
    elements.mosModelState.textContent = "Offline";
    elements.clipModelState.textContent = "Offline";
    elements.apiEndpointSignal.textContent = apiBase;
  }
}

async function updateMobileAccess() {
  const currentUrl = new URL(window.location.href);
  currentUrl.hash = "";
  let shareUrl = currentUrl.href;
  let helpText =
    "Scan from a device on the same Wi-Fi/network. This opens the demo page.";

  if (isLocalHost(currentUrl.hostname)) {
    try {
      const response = await fetch(
        `${getApiBase()}/access-info?frontend_port=${encodeURIComponent(currentUrl.port || "5173")}`,
      );
      if (response.ok) {
        const accessInfo = await response.json();
        if (accessInfo?.frontend_urls?.length) {
          shareUrl = accessInfo.frontend_urls[0];
          helpText = "Use this QR on a phone connected to the same Wi-Fi/network as this computer.";
        } else {
          helpText = "Could not detect a LAN address. Replace localhost with this computer's Wi-Fi IP.";
        }
      }
    } catch (error) {
      helpText = "Backend access-info is unavailable. Replace localhost with this computer's Wi-Fi IP.";
    }
  }

  setShareUrl(shareUrl, helpText);
}

function setShareUrl(shareUrl, helpText) {
  elements.shareUrlInput.value = shareUrl;
  elements.qrHelp.textContent = helpText;
  elements.qrFallback.hidden = true;
  elements.qrImage.hidden = false;
  elements.qrImage.src = buildQrImageUrl(shareUrl);
}

function buildQrImageUrl(value) {
  return `https://api.qrserver.com/v1/create-qr-code/?size=180x180&margin=10&data=${encodeURIComponent(value)}`;
}

async function copyShareUrl() {
  const shareUrl = elements.shareUrlInput.value;
  if (!shareUrl) return;

  try {
    await navigator.clipboard.writeText(shareUrl);
    elements.qrHelp.textContent = "Access URL copied. Open it on a device connected to the same network.";
  } catch (error) {
    elements.shareUrlInput.focus();
    elements.shareUrlInput.select();
    elements.qrHelp.textContent = "Copy failed automatically. Select the URL field and copy it manually.";
  }
}

function setBackendState(stateName, statusText, modelText) {
  elements.statusDot.className = stateName === "error" ? "status-dot error" : "status-dot connected";
  elements.statusText.textContent = statusText;
  elements.modelStatusText.textContent = modelText;
}

function setBusy(isBusy, message = "Checking listing...") {
  elements.analyzeButton.disabled = isBusy || !state.file;
  elements.analyzeButton.textContent = isBusy ? "Checking listing..." : "Run SmartQC Check";
  elements.loadingIndicator.hidden = !isBusy;
  elements.loadingIndicator.textContent = message;
}

function setDraftBusy(isBusy) {
  elements.saveDraftButton.disabled = isBusy;
  elements.saveDraftButton.textContent = isBusy ? "Saving..." : "Save Draft";
}

function showMessage(message, type = "error") {
  elements.errorMessage.hidden = !message;
  elements.errorMessage.textContent = message || "";
  elements.errorMessage.className = type === "notice" ? "error-message notice" : "error-message";
}

function hasProductText() {
  return Boolean(elements.productNameInput.value.trim() || elements.descriptionInput.value.trim());
}

function isLocalHost(hostname) {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
}

function normalizeApiBase(value) {
  return String(value || "").trim().replace(/\/$/, "");
}

function getApiBase() {
  return normalizeApiBase(elements.apiBaseInput.value || inferBackendBase());
}

function inferBackendBase() {
  const pageHost = window.location.hostname || "127.0.0.1";
  const hostname = isLocalHost(pageHost) ? "127.0.0.1" : pageHost;
  const protocol = window.location.protocol === "https:" ? "https:" : "http:";
  return `${protocol}//${hostname}:8000`;
}

function normalizeDecision(value) {
  return ["pass", "review", "reject"].includes(value) ? value : "review";
}

function decisionLabel(decision) {
  const labels = {
    pass: "Ready to publish",
    review: "Needs review",
    reject: "Not recommended",
  };
  return labels[decision] || "Needs review";
}

function stage1UserLabel(status) {
  const labels = {
    consistent: "Product image matches the listing",
    review: "Product match needs review",
    missing_text_review: "Product text missing",
    inconsistent: "Image may not match the listing",
    invalid_image: "Image may not be a valid product photo",
    ready_text_image: "Ready for product match check",
    ready_image_only: "Ready for image-only check",
    request_failed: "Request failed",
    not_run: "Not run",
  };
  return labels[status] || formatTechnicalStatus(status);
}

function stage2UserLabel(status) {
  const labels = {
    completed: "Completed",
    skipped: "Skipped",
    not_run: "Not run",
  };
  return labels[status] || formatTechnicalStatus(status);
}

function stage1StatusClass(status) {
  if (status === "consistent") return "status-positive";
  if (status === "review" || status === "missing_text_review") return "status-warning";
  if (status === "inconsistent" || status === "invalid_image" || status === "request_failed") {
    return "status-danger";
  }
  return "status-neutral";
}

function stage2StatusClass(status) {
  if (status === "completed") return "status-positive";
  if (status === "skipped") return "status-skipped";
  return "status-neutral";
}

function getQualityLabel(value) {
  if (!Number.isFinite(value)) return "N/A";
  if (value >= 70) return "Image quality looks good";
  if (value >= 50) return "Image quality is acceptable but could be improved";
  return "Image quality is low";
}

function mapRecommendation(recommendation, decision) {
  if (decision === "pass") return "This image looks suitable for your product listing.";
  if (decision === "review") return "Please review the image and product details before publishing.";
  if (decision === "reject") return "Please upload a clearer and more relevant product image.";
  return recommendation;
}

function formatTechnicalStatus(value) {
  if (value === null || value === undefined || value === "") return "N/A";
  return String(value).replaceAll("_", " ");
}

function formatRawPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "N/A";
  return `${(numeric * 100).toFixed(1)}%`;
}

function formatQualityScore(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "N/A";
  return `${numeric.toFixed(1)} / 100`;
}

function formatMosScore(value, maxScore = 4) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "N/A";
  const maxNumeric = Number(maxScore);
  const displayMax = Number.isFinite(maxNumeric) ? maxNumeric : 4;
  return `${numeric.toFixed(2)} / ${displayMax}`;
}

function scoreToPercent(value) {
  const numeric = Number(value);
  return clampPercent(Number.isFinite(numeric) ? numeric : 0);
}

function formatBoolean(value) {
  if (value === true) return "Yes";
  if (value === false) return "No";
  return "N/A";
}

function displayValue(value, fallback = "N/A") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function formatPreviewPrice(value) {
  const text = String(value || "").trim();
  if (!text) return "$0.00";
  if (text.startsWith("$")) return text;
  const numeric = Number(text);
  return Number.isFinite(numeric) ? `$${numeric.toFixed(2)}` : text;
}

function formatDraftDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown time";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function truncateText(value, length) {
  const text = String(value || "").trim();
  return text.length > length ? `${text.slice(0, length - 3)}...` : text;
}

function countWords(value) {
  return String(value || "").trim().split(/\s+/).filter(Boolean).length;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function clampPercent(value) {
  return clamp(Number.isFinite(value) ? value : 0, 0, 100);
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

resetResults();
updateTextState();
updateListingPreview();
updateConnectionStatus();
updateMobileAccess();
loadDrafts();
