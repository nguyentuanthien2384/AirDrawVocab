const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");
const brushSize = document.getElementById("brushSize");
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const loading = document.getElementById("loading");
const emptyState = document.getElementById("emptyState");
const resultBox = document.getElementById("resultBox");
const predictedLabel = document.getElementById("predictedLabel");
const confidenceText = document.getElementById("confidenceText");
const top3List = document.getElementById("top3List");
const chatbotReply = document.getElementById("chatbotReply");
const enhancedDrawingImg = document.getElementById("enhancedDrawingImg");
const generateImageBtn = document.getElementById("generateImageBtn");
const referenceImage = document.getElementById("referenceImage");
const referencePlaceholder = document.getElementById("referencePlaceholder");
const referenceStatus = document.getElementById("referenceStatus");
const faceVideo = document.getElementById("faceVideo");
const faceCanvas = document.getElementById("faceCanvas");
const targetGuideCanvas = document.getElementById("targetGuideCanvas");
const targetGuideCtx = targetGuideCanvas.getContext("2d");
const airDrawingCanvas = document.getElementById("airDrawingCanvas");
const airDrawingCtx = airDrawingCanvas.getContext("2d");
const handOverlayCanvas = document.getElementById("handOverlayCanvas");
const handOverlayCtx = handOverlayCanvas.getContext("2d");
const faceUsername = document.getElementById("faceUsername");
const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const enrollFaceBtn = document.getElementById("enrollFaceBtn");
const verifyFaceBtn = document.getElementById("verifyFaceBtn");
const faceStatus = document.getElementById("faceStatus");
const startHandDrawBtn = document.getElementById("startHandDrawBtn");
const stopHandDrawBtn = document.getElementById("stopHandDrawBtn");
const handDrawStatus = document.getElementById("handDrawStatus");
const cameraBrushSize = document.getElementById("cameraBrushSize");
const handSmoothness = document.getElementById("handSmoothness");
const gameTarget = document.getElementById("gameTarget");
const gameTime = document.getElementById("gameTime");
const gameLevel = document.getElementById("gameLevel");
const gameLives = document.getElementById("gameLives");
const gameScore = document.getElementById("gameScore");
const gameStreak = document.getElementById("gameStreak");
const startQuickDrawBtn = document.getElementById("startQuickDrawBtn");
const skipRoundBtn = document.getElementById("skipRoundBtn");
const clearAirDrawBtn = document.getElementById("clearAirDrawBtn");
const submitAirDrawBtn = document.getElementById("submitAirDrawBtn");
const showGuideToggle = document.getElementById("showGuideToggle");
const quickDrawStatus = document.getElementById("quickDrawStatus");
const quickDrawGuesses = document.getElementById("quickDrawGuesses");
const targetGuideCard = document.getElementById("targetGuideCard");
const targetGuideLabel = document.getElementById("targetGuideLabel");
const targetGuideHint = document.getElementById("targetGuideHint");
const targetGuidePreview = document.getElementById("targetGuidePreview");
const targetGuidePreviewCtx = targetGuidePreview.getContext("2d");
const recognizedCard = document.getElementById("recognizedCard");
const recognizedLabel = document.getElementById("recognizedLabel");
const recognizedConfidence = document.getElementById("recognizedConfidence");
const recognizedImage = document.getElementById("recognizedImage");
const cameraPredictionOverlay = document.getElementById("cameraPredictionOverlay");
const cameraPredictionLabel = document.getElementById("cameraPredictionLabel");
const cameraObjectImage = document.getElementById("cameraObjectImage");
let faceStream = null;
let faceVerified = false;

let isDrawing = false;
let hasDrawn = false;
let currentLabel = "";
let handModel = null;
let handDrawingActive = false;
let handLoopRunning = false;
let lastHandPoint = null;
let lastCameraPoint = null;
let filteredHandPoint = null;
let filteredCameraPoint = null;
let lastHandClearAt = 0;
let lastHandStatus = "";
let lastDrawingGestureAt = 0;
let gameTimerId = null;
let gameGuessTimerId = null;
let gamePredictionInFlight = false;
let lastPreviewLabel = "";
const referenceImageCache = new Map();
const referenceImageRequests = new Map();

const QUICKDRAW_LABELS = [
  "apple",
  "baseball",
  "book",
  "bowtie",
  "diamond",
  "dog",
  "door",
  "envelope",
  "eye",
  "fish",
  "hat",
  "leaf",
  "lightning",
  "moon",
  "pants",
  "scissors",
  "square",
  "star",
  "t-shirt",
];

const DRAWING_HINTS = {
  apple: "Vẽ vòng tròn, thêm cuống và lá nhỏ phía trên.",
  baseball: "Vẽ vòng tròn lớn, thêm 2 đường cong khâu bóng ở hai bên.",
  book: "Vẽ hình chữ nhật, kẻ gáy sách ở giữa.",
  bowtie: "Vẽ 2 tam giác chạm nhau, thêm nút nhỏ ở giữa.",
  diamond: "Vẽ hình thoi: đỉnh trên, đỉnh dưới và hai góc ngang.",
  dog: "Vẽ đầu tròn, 2 tai, mắt và mũi đơn giản.",
  door: "Vẽ hình chữ nhật đứng, thêm tay nắm tròn.",
  envelope: "Vẽ hình chữ nhật ngang, thêm nét chữ V như phong bì.",
  eye: "Vẽ oval nằm ngang, thêm tròng mắt và con ngươi.",
  fish: "Vẽ thân oval, đuôi tam giác và mắt nhỏ.",
  hat: "Vẽ nửa vòng tròn của mũ, thêm vành ngang phía dưới.",
  leaf: "Vẽ oval nhọn, thêm gân lá ở giữa.",
  lightning: "Vẽ đường zigzag nhọn từ trên xuống.",
  moon: "Vẽ trăng lưỡi liềm bằng 2 đường cong.",
  pants: "Vẽ cạp quần và 2 ống quần tách nhau.",
  scissors: "Vẽ chữ X trước, thêm 2 vòng tròn tay cầm ở một bên.",
  square: "Vẽ 4 cạnh đều, khép kín thành hình vuông.",
  star: "Vẽ 5 đỉnh nhọn nối liên tục.",
  "t-shirt": "Vẽ thân áo, 2 tay áo và cổ áo.",
};

const GUIDE_DRAWERS = {
  apple: (g) => {
    g.ellipse(0.5, 0.55, 0.22, 0.25);
    g.line(0.5, 0.3, 0.5, 0.2);
    g.ellipse(0.6, 0.24, 0.09, 0.04);
  },
  baseball: (g) => {
    g.circle(0.5, 0.5, 0.28);
    g.ellipse(0.34, 0.5, 0.1, 0.28, -1.2, 1.2);
    g.ellipse(0.66, 0.5, 0.1, 0.28, 1.9, 4.4);
  },
  book: (g) => {
    g.rect(0.22, 0.28, 0.56, 0.44);
    g.line(0.5, 0.28, 0.5, 0.72);
    g.line(0.28, 0.38, 0.44, 0.38);
    g.line(0.56, 0.38, 0.72, 0.38);
  },
  bowtie: (g) => {
    g.polygon([[0.18, 0.35], [0.45, 0.5], [0.18, 0.65]]);
    g.polygon([[0.82, 0.35], [0.55, 0.5], [0.82, 0.65]]);
    g.rect(0.45, 0.42, 0.1, 0.16);
  },
  diamond: (g) => {
    g.polygon([[0.5, 0.15], [0.78, 0.48], [0.5, 0.85], [0.22, 0.48]]);
    g.line(0.22, 0.48, 0.78, 0.48);
  },
  dog: (g) => {
    g.circle(0.5, 0.5, 0.24);
    g.ellipse(0.3, 0.42, 0.09, 0.16);
    g.ellipse(0.7, 0.42, 0.09, 0.16);
    g.circle(0.42, 0.48, 0.025);
    g.circle(0.58, 0.48, 0.025);
    g.circle(0.5, 0.57, 0.04);
  },
  door: (g) => {
    g.rect(0.32, 0.18, 0.36, 0.64);
    g.circle(0.6, 0.5, 0.025);
  },
  envelope: (g) => {
    g.rect(0.18, 0.33, 0.64, 0.34);
    g.line(0.18, 0.33, 0.5, 0.55);
    g.line(0.82, 0.33, 0.5, 0.55);
    g.line(0.18, 0.67, 0.42, 0.5);
    g.line(0.82, 0.67, 0.58, 0.5);
  },
  eye: (g) => {
    g.ellipse(0.5, 0.5, 0.34, 0.15);
    g.circle(0.5, 0.5, 0.1);
    g.circle(0.5, 0.5, 0.04);
  },
  fish: (g) => {
    g.ellipse(0.43, 0.5, 0.25, 0.16);
    g.polygon([[0.66, 0.5], [0.84, 0.34], [0.84, 0.66]]);
    g.circle(0.32, 0.46, 0.025);
  },
  hat: (g) => {
    g.ellipse(0.46, 0.55, 0.24, 0.16, Math.PI, Math.PI * 2);
    g.line(0.2, 0.58, 0.8, 0.58);
    g.line(0.28, 0.66, 0.72, 0.66);
  },
  leaf: (g) => {
    g.ellipse(0.5, 0.5, 0.17, 0.32, -0.45, Math.PI * 2 - 0.45);
    g.line(0.38, 0.76, 0.62, 0.24);
  },
  lightning: (g) => {
    g.path([[0.6, 0.15], [0.34, 0.5], [0.5, 0.5], [0.4, 0.85], [0.72, 0.42], [0.55, 0.42]]);
  },
  moon: (g) => {
    g.ellipse(0.48, 0.5, 0.24, 0.32, 0.75, Math.PI * 1.65);
    g.ellipse(0.58, 0.5, 0.18, 0.28, 0.78, Math.PI * 1.68);
  },
  pants: (g) => {
    g.path([[0.32, 0.18], [0.68, 0.18], [0.76, 0.82], [0.56, 0.82], [0.5, 0.45], [0.44, 0.82], [0.24, 0.82], [0.32, 0.18]]);
    g.line(0.5, 0.45, 0.5, 0.82);
  },
  scissors: (g) => {
    g.circle(0.3, 0.38, 0.1);
    g.circle(0.3, 0.66, 0.1);
    g.line(0.38, 0.52, 0.76, 0.24);
    g.line(0.38, 0.52, 0.78, 0.8);
    g.line(0.36, 0.43, 0.68, 0.74);
    g.line(0.36, 0.61, 0.68, 0.3);
  },
  square: (g) => {
    g.rect(0.28, 0.28, 0.44, 0.44);
  },
  star: (g) => {
    const points = [];
    for (let i = 0; i < 10; i += 1) {
      const angle = -Math.PI / 2 + (i * Math.PI) / 5;
      const radius = i % 2 === 0 ? 0.32 : 0.14;
      points.push([0.5 + Math.cos(angle) * radius, 0.52 + Math.sin(angle) * radius]);
    }
    g.polygon(points);
  },
  "t-shirt": (g) => {
    g.polygon([[0.34, 0.18], [0.43, 0.16], [0.5, 0.25], [0.57, 0.16], [0.66, 0.18], [0.82, 0.38], [0.68, 0.48], [0.68, 0.82], [0.32, 0.82], [0.32, 0.48], [0.18, 0.38]]);
    g.ellipse(0.5, 0.24, 0.08, 0.04, 0, Math.PI);
  },
};

const gameState = {
  active: false,
  roundActive: false,
  level: 1,
  maxLevels: 6,
  lives: 3,
  maxLives: 3,
  score: 0,
  streak: 0,
  timeLeft: 20,
  target: "",
};

async function startFaceCamera() {
  if (faceStream) {
    await waitForFaceVideoReady();
    startCameraBtn.disabled = true;
    stopCameraBtn.disabled = false;
    return faceStream;
  }

  try {
    faceStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user",
      },
      audio: false,
    });
    faceVideo.srcObject = faceStream;
    await waitForFaceVideoReady();
    faceStatus.textContent = "Camera đã bật.";
    startCameraBtn.disabled = true;
    stopCameraBtn.disabled = false;
    return faceStream;
  } catch (error) {
    faceStatus.textContent = `Không mở được camera: ${error.message}`;
    throw error;
  }
}

function stopFaceCamera() {
  stopHandDrawing();
  clearGameTimers();
  gameState.active = false;
  gameState.roundActive = false;
  updateGameHud();
  setQuickDrawStatus("Đã tắt camera.");

  if (faceStream) {
    faceStream.getTracks().forEach((track) => track.stop());
    faceStream = null;
  }

  faceVideo.pause();
  faceVideo.removeAttribute("srcObject");
  faceVideo.srcObject = null;
  clearHandOverlay();
  resetHandStroke();
  startCameraBtn.disabled = false;
  stopCameraBtn.disabled = true;
  faceStatus.textContent = "Camera đã tắt.";
}

function waitForFaceVideoReady() {
  if (faceVideo.readyState >= 2 && faceVideo.videoWidth) {
    syncHandOverlaySize();
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    const done = () => {
      faceVideo.removeEventListener("loadedmetadata", done);
      faceVideo.removeEventListener("canplay", done);
      faceVideo.play().catch(() => {});
      syncHandOverlaySize();
      resolve();
    };
    faceVideo.addEventListener("loadedmetadata", done, { once: true });
    faceVideo.addEventListener("canplay", done, { once: true });
  });
}

function syncHandOverlaySize() {
  const width = faceVideo.videoWidth || 640;
  const height = faceVideo.videoHeight || 480;
  if (targetGuideCanvas.width !== width || targetGuideCanvas.height !== height) {
    targetGuideCanvas.width = width;
    targetGuideCanvas.height = height;
  }
  if (handOverlayCanvas.width !== width || handOverlayCanvas.height !== height) {
    handOverlayCanvas.width = width;
    handOverlayCanvas.height = height;
  }
  if (airDrawingCanvas.width !== width || airDrawingCanvas.height !== height) {
    airDrawingCanvas.width = width;
    airDrawingCanvas.height = height;
  }
  renderTargetGuide();
}

function createGuideApi(targetCtx, mode = "overlay") {
  const width = targetCtx.canvas.width;
  const height = targetCtx.canvas.height;
  const size = Math.min(width, height) * (mode === "overlay" ? 0.72 : 0.78);
  const offsetX = (width - size) / 2;
  const offsetY = (height - size) / 2;

  const px = (x) => offsetX + x * size;
  const py = (y) => offsetY + y * size;

  function path(points, close = false) {
    if (!points.length) return;
    targetCtx.beginPath();
    targetCtx.moveTo(px(points[0][0]), py(points[0][1]));
    points.slice(1).forEach(([x, y]) => targetCtx.lineTo(px(x), py(y)));
    if (close) targetCtx.closePath();
    targetCtx.stroke();
  }

  return {
    path,
    line: (x1, y1, x2, y2) => path([[x1, y1], [x2, y2]]),
    polygon: (points) => path(points, true),
    rect: (x, y, w, h) => targetCtx.strokeRect(px(x), py(y), w * size, h * size),
    circle: (x, y, r) => {
      targetCtx.beginPath();
      targetCtx.arc(px(x), py(y), r * size, 0, Math.PI * 2);
      targetCtx.stroke();
    },
    ellipse: (x, y, rx, ry, start = 0, end = Math.PI * 2) => {
      targetCtx.beginPath();
      targetCtx.ellipse(px(x), py(y), rx * size, ry * size, 0, start, end);
      targetCtx.stroke();
    },
  };
}

function renderGuideShape(label, targetCtx, mode = "overlay") {
  targetCtx.clearRect(0, 0, targetCtx.canvas.width, targetCtx.canvas.height);
  if (!label) return;

  const drawer = GUIDE_DRAWERS[label];
  if (!drawer) return;

  targetCtx.save();
  targetCtx.lineCap = "round";
  targetCtx.lineJoin = "round";
  targetCtx.lineWidth = Math.max(3, Math.min(targetCtx.canvas.width, targetCtx.canvas.height) * 0.012);
  targetCtx.strokeStyle = mode === "overlay" ? "rgba(255, 255, 255, 0.82)" : "#17355f";
  targetCtx.shadowColor = mode === "overlay" ? "rgba(15, 23, 42, 0.75)" : "rgba(37, 99, 235, 0.18)";
  targetCtx.shadowBlur = mode === "overlay" ? 6 : 3;
  drawer(createGuideApi(targetCtx, mode));
  targetCtx.restore();
}

function renderTargetGuide() {
  const label = gameState.target;
  const showOverlay = Boolean(gameState.roundActive && label && showGuideToggle.checked);
  targetGuideCanvas.classList.toggle("hidden", !showOverlay);
  targetGuideCard.classList.toggle("hidden", !gameState.active || !label);
  targetGuideLabel.textContent = label || "---";
  targetGuideHint.textContent = label
    ? DRAWING_HINTS[label] || `Vẽ ${label} bằng các nét đơn giản.`
    : "Bắt đầu game để xem mẫu.";
  renderGuideShape(showOverlay ? label : "", targetGuideCtx, "overlay");
  renderGuideShape(gameState.active && label ? label : "", targetGuidePreviewCtx, "preview");
}

function markCanvasChanged() {
  const hadResult = currentLabel || !resultBox.classList.contains("hidden");
  hasDrawn = true;
  if (hadResult) {
    currentLabel = "";
    resultBox.classList.add("hidden");
    emptyState.classList.remove("hidden");
    resetReferenceImage();
  }
}

async function captureFaceBlob() {
  if (!faceStream) {
    await startFaceCamera();
  }
  if (!faceVideo.videoWidth) {
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  const ctxFace = faceCanvas.getContext("2d");
  ctxFace.drawImage(faceVideo, 0, 0, faceCanvas.width, faceCanvas.height);
  return new Promise((resolve) => faceCanvas.toBlob(resolve, "image/png"));
}

async function sendFace(mode) {
  const username = faceUsername.value.trim();
  if (mode === "enroll" && !username) {
    alert("Bạn cần nhập tên người dùng trước khi đăng ký khuôn mặt.");
    return;
  }
  faceStatus.textContent = mode === "enroll" ? "Đang đăng ký khuôn mặt..." : "Đang xác thực khuôn mặt...";
  try {
    const blob = await captureFaceBlob();
    const formData = new FormData();
    formData.append("file", blob, "face.png");
    formData.append("username", username);
    const response = await fetch(`/face/${mode}`, { method: "POST", body: formData });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.detail || "Không gọi được API khuôn mặt.");
    faceVerified = Boolean(result.ok);
    faceStatus.textContent = result.message || (result.ok ? "Thành công" : "Thất bại");
  } catch (error) {
    faceVerified = false;
    faceStatus.textContent = `Lỗi khuôn mặt: ${error.message}`;
  }
}

function resetCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = Number(brushSize.value);
  hasDrawn = false;
  resetHandStroke();
  clearAirDrawingOverlay();
}

function getPointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  const clientX = event.touches ? event.touches[0].clientX : event.clientX;
  const clientY = event.touches ? event.touches[0].clientY : event.clientY;
  return {
    x: ((clientX - rect.left) / rect.width) * canvas.width,
    y: ((clientY - rect.top) / rect.height) * canvas.height,
  };
}

function startDrawing(event) {
  event.preventDefault();
  isDrawing = true;
  markCanvasChanged();
  const pos = getPointerPosition(event);
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
}

function draw(event) {
  if (!isDrawing) return;
  event.preventDefault();
  const pos = getPointerPosition(event);
  ctx.lineWidth = Number(brushSize.value);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
}

function stopDrawing(event) {
  if (!isDrawing) return;
  event.preventDefault();
  isDrawing = false;
  ctx.closePath();
}

function setHandStatus(message) {
  if (lastHandStatus === message) return;
  lastHandStatus = message;
  handDrawStatus.textContent = message;
}

function setHandDrawingControls(active) {
  startHandDrawBtn.disabled = active;
  stopHandDrawBtn.disabled = !active;
}

function clearHandOverlay() {
  handOverlayCtx.clearRect(0, 0, handOverlayCanvas.width, handOverlayCanvas.height);
}

function clearAirDrawingOverlay() {
  airDrawingCtx.clearRect(0, 0, airDrawingCanvas.width, airDrawingCanvas.height);
}

function isFingerUp(landmarks, tipIndex, pipIndex) {
  return landmarks[tipIndex].y < landmarks[pipIndex].y - 0.015;
}

function resetHandStroke() {
  lastHandPoint = null;
  lastCameraPoint = null;
  filteredHandPoint = null;
  filteredCameraPoint = null;
}

function smoothPoint(point, previous) {
  if (!previous) return point;
  const keepPrevious = Number(handSmoothness.value || 76) / 100;
  const keepCurrent = 1 - keepPrevious;
  return {
    x: previous.x * keepPrevious + point.x * keepCurrent,
    y: previous.y * keepPrevious + point.y * keepCurrent,
  };
}

function drawRoundDot(targetCtx, point, radius, color) {
  targetCtx.beginPath();
  targetCtx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  targetCtx.fillStyle = color;
  targetCtx.fill();
}

function drawCameraPoint(x, y) {
  airDrawingCtx.lineCap = "round";
  airDrawingCtx.lineJoin = "round";
  airDrawingCtx.strokeStyle = "#22d3ee";
  airDrawingCtx.fillStyle = "#22d3ee";
  airDrawingCtx.lineWidth = Number(cameraBrushSize.value || 4);
  airDrawingCtx.shadowColor = "rgba(34, 211, 238, 0.35)";
  airDrawingCtx.shadowBlur = 2;
  const current = { x, y };

  if (!lastCameraPoint) {
    drawRoundDot(airDrawingCtx, current, airDrawingCtx.lineWidth / 2, "#22d3ee");
  } else {
    const dx = current.x - lastCameraPoint.x;
    const dy = current.y - lastCameraPoint.y;
    const distance = Math.hypot(dx, dy);
    airDrawingCtx.beginPath();
    if (distance > Math.max(airDrawingCanvas.width, airDrawingCanvas.height) * 0.45) {
      drawRoundDot(airDrawingCtx, current, airDrawingCtx.lineWidth / 2, "#22d3ee");
    } else {
      const midX = (lastCameraPoint.x + current.x) / 2;
      const midY = (lastCameraPoint.y + current.y) / 2;
      airDrawingCtx.moveTo(lastCameraPoint.x, lastCameraPoint.y);
      airDrawingCtx.quadraticCurveTo(lastCameraPoint.x, lastCameraPoint.y, midX, midY);
      airDrawingCtx.stroke();
    }
  }

  airDrawingCtx.shadowBlur = 0;
  lastCameraPoint = current;
}

function drawHandPoint(x, y, cameraX, cameraY) {
  markCanvasChanged();
  ctx.lineWidth = Math.max(10, Number(brushSize.value));
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#000000";
  const current = { x, y };

  if (!lastHandPoint) {
    drawRoundDot(ctx, current, ctx.lineWidth / 2, "#000000");
  } else {
    const dx = current.x - lastHandPoint.x;
    const dy = current.y - lastHandPoint.y;
    const distance = Math.hypot(dx, dy);
    ctx.beginPath();
    if (distance > Math.max(canvas.width, canvas.height) * 0.45) {
      drawRoundDot(ctx, current, ctx.lineWidth / 2, "#000000");
    } else {
      const midX = (lastHandPoint.x + current.x) / 2;
      const midY = (lastHandPoint.y + current.y) / 2;
      ctx.moveTo(lastHandPoint.x, lastHandPoint.y);
      ctx.quadraticCurveTo(lastHandPoint.x, lastHandPoint.y, midX, midY);
      ctx.stroke();
    }
  }

  lastHandPoint = current;
  drawCameraPoint(cameraX, cameraY);
}

function handleHandResults(results) {
  syncHandOverlaySize();
  clearHandOverlay();

  const landmarks = results.multiHandLandmarks && results.multiHandLandmarks[0];
  if (!landmarks) {
    resetHandStroke();
    setHandStatus("Đang tìm bàn tay...");
    return;
  }

  if (window.drawConnectors && window.drawLandmarks && window.HAND_CONNECTIONS) {
    window.drawConnectors(handOverlayCtx, landmarks, window.HAND_CONNECTIONS, {
      color: "#22c55e",
      lineWidth: 3,
    });
    window.drawLandmarks(handOverlayCtx, landmarks, {
      color: "#38bdf8",
      lineWidth: 1,
      radius: 3,
    });
  }

  const indexUp = isFingerUp(landmarks, 8, 6);
  const middleUp = isFingerUp(landmarks, 12, 10);
  const ringUp = isFingerUp(landmarks, 16, 14);
  const pinkyUp = isFingerUp(landmarks, 20, 18);
  const openPalm = indexUp && middleUp && ringUp && pinkyUp;
  const drawingGesture = indexUp && !middleUp;
  const now = Date.now();

  if (openPalm) {
    resetHandStroke();
    if (Date.now() - lastHandClearAt > 1200) {
      resetCanvas();
      currentLabel = "";
      emptyState.classList.remove("hidden");
      resultBox.classList.add("hidden");
      resetReferenceImage();
      lastHandClearAt = Date.now();
    }
    setHandStatus("Đã xóa bảng vẽ.");
    return;
  }

  if (!drawingGesture) {
    if (now - lastDrawingGestureAt > 220) {
      resetHandStroke();
    }
    setHandStatus("Sẵn sàng bắt nét ngón trỏ.");
    return;
  }

  lastDrawingGestureAt = now;

  const tip = landmarks[8];
  const rawHandPoint = {
    x: (1 - tip.x) * canvas.width,
    y: tip.y * canvas.height,
  };
  const rawCameraPoint = {
    x: tip.x * airDrawingCanvas.width,
    y: tip.y * airDrawingCanvas.height,
  };

  if (
    filteredHandPoint &&
    Math.hypot(rawHandPoint.x - filteredHandPoint.x, rawHandPoint.y - filteredHandPoint.y) >
      Math.max(canvas.width, canvas.height) * 0.55
  ) {
    resetHandStroke();
  }

  filteredHandPoint = smoothPoint(rawHandPoint, filteredHandPoint);
  filteredCameraPoint = smoothPoint(rawCameraPoint, filteredCameraPoint);
  drawHandPoint(
    filteredHandPoint.x,
    filteredHandPoint.y,
    filteredCameraPoint.x,
    filteredCameraPoint.y
  );
  setHandStatus("Đang vẽ bằng tay.");
}

async function ensureHandModel() {
  if (handModel) return handModel;

  if (!window.Hands) {
    throw new Error("MediaPipe Hands chưa tải được. Hãy kiểm tra internet rồi tải lại trang.");
  }

  handModel = new window.Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });
  handModel.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.72,
    minTrackingConfidence: 0.55,
  });
  handModel.onResults(handleHandResults);
  return handModel;
}

async function runHandDrawingLoop() {
  if (handLoopRunning) return;
  handLoopRunning = true;

  while (handDrawingActive) {
    try {
      await handModel.send({ image: faceVideo });
    } catch (error) {
      handDrawingActive = false;
      setHandDrawingControls(false);
      setHandStatus(`Lỗi vẽ tay: ${error.message}`);
      break;
    }
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }

  handLoopRunning = false;
  resetHandStroke();
  clearHandOverlay();
}

async function startHandDrawing() {
  setHandStatus("Đang khởi động vẽ tay...");
  try {
    await startFaceCamera();
    await ensureHandModel();
    handDrawingActive = true;
    setHandDrawingControls(true);
    setHandStatus("Đang tìm bàn tay...");
    runHandDrawingLoop();
  } catch (error) {
    handDrawingActive = false;
    setHandDrawingControls(false);
    setHandStatus(`Không bật được vẽ tay: ${error.message}`);
  }
}

function stopHandDrawing() {
  handDrawingActive = false;
  resetHandStroke();
  setHandDrawingControls(false);
  setHandStatus("Đã dừng vẽ tay.");
}

function updateGameHud() {
  gameTarget.textContent = gameState.target || "---";
  gameTime.textContent = String(gameState.timeLeft).padStart(2, "0");
  gameLevel.textContent = `${gameState.level}/${gameState.maxLevels}`;
  gameLives.textContent =
    "♥".repeat(Math.max(gameState.lives, 0)) +
    "♡".repeat(Math.max(gameState.maxLives - gameState.lives, 0));
  gameScore.textContent = String(gameState.score);
  gameStreak.textContent = String(gameState.streak);
  skipRoundBtn.disabled = !gameState.roundActive;
  renderTargetGuide();
}

function setQuickDrawStatus(message) {
  quickDrawStatus.textContent = message;
}

function setQuickDrawGuesses(items = []) {
  if (!items.length) {
    quickDrawGuesses.textContent = "AI chưa đoán.";
    return;
  }

  quickDrawGuesses.innerHTML = "";
  items.slice(0, 3).forEach((item) => {
    const row = document.createElement("div");
    row.className = "guess-row";
    row.innerHTML = `
      <span>${item.label}</span>
      <strong>${(item.confidence * 100).toFixed(1)}%</strong>
    `;
    quickDrawGuesses.appendChild(row);
  });
}

function clearRecognizedPreview() {
  lastPreviewLabel = "";
  recognizedLabel.textContent = "---";
  recognizedConfidence.textContent = "Vẽ bằng ngón trỏ để hiện ảnh tương ứng.";
  recognizedImage.removeAttribute("src");
  cameraPredictionOverlay.classList.add("hidden");
  cameraPredictionLabel.textContent = "---";
  cameraObjectImage.classList.add("hidden");
  cameraObjectImage.removeAttribute("src");
}

async function getOfflineReferenceImage(label) {
  if (referenceImageCache.has(label)) {
    return referenceImageCache.get(label);
  }
  if (referenceImageRequests.has(label)) {
    return referenceImageRequests.get(label);
  }

  const request = (async () => {
    const formData = new FormData();
    formData.append("label", label);
    const response = await fetch("/image/reference", {
      method: "POST",
      body: formData,
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(result.detail || "Không tải được ảnh minh họa.");
    }
    referenceImageCache.set(label, result.image);
    referenceImageRequests.delete(label);
    return result.image;
  })();

  referenceImageRequests.set(label, request);
  return request;
}

async function updateRecognizedPreview(result) {
  if (!result || !result.label) return;

  const confidence = Number(result.confidence || 0);
  const label = result.label;
  const percent = result.confidence_percent ?? (confidence * 100).toFixed(1);
  const labelText = confidence < 0.25 ? `${label} ?` : label;

  recognizedLabel.textContent = labelText;
  recognizedConfidence.textContent = `Độ tin cậy: ${percent}%`;
  cameraPredictionLabel.textContent = `${labelText} · ${percent}%`;
  cameraPredictionOverlay.classList.remove("hidden");

  if (lastPreviewLabel === label && recognizedImage.getAttribute("src")) return;
  lastPreviewLabel = label;

  try {
    const image = await getOfflineReferenceImage(label);
    if (lastPreviewLabel !== label) return;
    recognizedImage.src = image;
    cameraObjectImage.src = image;
    cameraObjectImage.classList.remove("hidden");
  } catch (error) {
    recognizedConfidence.textContent = `Đã nhận diện ${label}, nhưng chưa tải được ảnh minh họa.`;
  }
}

function pickQuickDrawTarget() {
  const candidates = QUICKDRAW_LABELS.filter((label) => label !== gameState.target);
  return candidates[Math.floor(Math.random() * candidates.length)];
}

function clearGameTimers() {
  if (gameTimerId) {
    clearInterval(gameTimerId);
    gameTimerId = null;
  }
  if (gameGuessTimerId) {
    clearInterval(gameGuessTimerId);
    gameGuessTimerId = null;
  }
}

function resetDrawingForRound() {
  resetCanvas();
  currentLabel = "";
  setQuickDrawGuesses();
  clearRecognizedPreview();
  resultBox.classList.add("hidden");
  emptyState.classList.remove("hidden");
  resetReferenceImage();
}

async function requestCanvasPrediction(source = "canvas") {
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  const formData = new FormData();
  formData.append("file", blob, "drawing.png");
  formData.append("source", source);

  const response = await fetch("/predict", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Không gọi được API nhận diện.");
  }

  return response.json();
}

function finishCurrentRound() {
  gameState.roundActive = false;
  clearGameTimers();
  resetHandStroke();
  updateGameHud();
}

function startRoundTimers() {
  clearGameTimers();

  gameTimerId = setInterval(() => {
    if (!gameState.roundActive) return;
    gameState.timeLeft -= 1;
    updateGameHud();
    if (gameState.timeLeft <= 0) {
      handleRoundTimeout();
    }
  }, 1000);

  gameGuessTimerId = setInterval(() => {
    predictForQuickDraw(false);
  }, 1100);
}

async function startQuickDrawRound() {
  await startHandDrawing();
  resetDrawingForRound();
  gameState.target = pickQuickDrawTarget();
  gameState.timeLeft = 20;
  gameState.roundActive = true;
  updateGameHud();
  setQuickDrawStatus(`Draw ${gameState.target} in under 20 seconds.`);
  startRoundTimers();
}

async function startQuickDrawGame() {
  gameState.active = true;
  gameState.roundActive = false;
  gameState.level = 1;
  gameState.lives = gameState.maxLives;
  gameState.score = 0;
  gameState.streak = 0;
  gameState.target = "";
  startQuickDrawBtn.textContent = "Chơi lại";
  updateGameHud();
  try {
    await startQuickDrawRound();
  } catch (error) {
    gameState.active = false;
    gameState.roundActive = false;
    updateGameHud();
    setQuickDrawStatus(`Không bắt đầu được: ${error.message}`);
  }
}

function advanceQuickDrawRound(delayMs = 900) {
  setTimeout(() => {
    if (!gameState.active) return;
    if (gameState.level >= gameState.maxLevels || gameState.lives <= 0) {
      endQuickDrawGame();
      return;
    }
    gameState.level += 1;
    startQuickDrawRound();
  }, delayMs);
}

function handleRoundSuccess(result) {
  if (!gameState.roundActive) return;
  finishCurrentRound();
  const bonus = 100 + gameState.timeLeft * 5 + gameState.streak * 20;
  gameState.score += bonus;
  gameState.streak += 1;
  currentLabel = result.label;
  updateGameHud();
  setQuickDrawStatus(`Great job! AI recognized ${result.label}. +${bonus} points.`);
  advanceQuickDrawRound();
}

function handleRoundTimeout() {
  if (!gameState.roundActive) return;
  finishCurrentRound();
  gameState.lives -= 1;
  gameState.streak = 0;
  updateGameHud();
  setQuickDrawStatus(`Time's up. The word was ${gameState.target}.`);
  advanceQuickDrawRound();
}

function skipQuickDrawRound() {
  if (!gameState.roundActive) return;
  finishCurrentRound();
  gameState.streak = 0;
  updateGameHud();
  setQuickDrawStatus("Try this one instead.");
  advanceQuickDrawRound(350);
}

function endQuickDrawGame() {
  finishCurrentRound();
  gameState.active = false;
  const completed = gameState.lives > 0 && gameState.level >= gameState.maxLevels;
  setQuickDrawStatus(
    completed
      ? `Hoàn thành! Score: ${gameState.score}.`
      : `Game over. Score: ${gameState.score}.`
  );
  setQuickDrawGuesses();
}

async function predictForQuickDraw(manual = false) {
  if (!hasDrawn) {
    if (manual) setQuickDrawStatus("Bạn cần vẽ trước khi nhận diện.");
    return;
  }
  if (gamePredictionInFlight) return;

  gamePredictionInFlight = true;
  try {
    const result = await requestCanvasPrediction("camera");
    setQuickDrawGuesses(result.top3 || []);
    updateRecognizedPreview(result);
    const topGuess = result.label ? `${result.label} (${result.confidence_percent}%)` : "chưa rõ";

    if (!gameState.roundActive) {
      setQuickDrawStatus(`AI thinks: ${topGuess}.`);
      return;
    }

    if (result.label === gameState.target && Number(result.confidence || 0) >= 0.45) {
      handleRoundSuccess(result);
    } else if (manual) {
      setQuickDrawStatus(`AI thinks: ${topGuess}. Hãy vẽ rõ hơn hoặc thử lại.`);
    } else {
      setQuickDrawStatus(`AI thinks: ${topGuess}.`);
    }
  } catch (error) {
    setQuickDrawStatus(`Lỗi nhận diện: ${error.message}`);
  } finally {
    gamePredictionInFlight = false;
  }
}

function setLoading(value) {
  loading.classList.toggle("hidden", !value);
  predictBtn.disabled = value;
}

function resetReferenceImage(message = "Chưa tạo ảnh tham khảo.") {
  referenceImage.classList.add("hidden");
  referenceImage.removeAttribute("src");
  referencePlaceholder.classList.remove("hidden");
  referencePlaceholder.textContent = message;
  referenceStatus.textContent = "";
}

function markdownLite(text) {
  return String(text)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>");
}

async function predictDrawing() {
  if (!hasDrawn) {
    alert("Bạn hãy vẽ một hình trước khi nhận diện nhé.");
    return;
  }

  setLoading(true);
  emptyState.classList.add("hidden");
  resultBox.classList.add("hidden");

  try {
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    const formData = new FormData();
    formData.append("file", blob, "drawing.png");

    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Không gọi được API nhận diện.");
    }

    const result = await response.json();
    currentLabel = result.label;
    updateRecognizedPreview(result);
    predictedLabel.textContent = `${result.label} (${result.meaning_vi})`;
    confidenceText.textContent = `Độ tin cậy: ${result.confidence_percent}%`;
    enhancedDrawingImg.src = result.enhanced_drawing;
    resetReferenceImage("Bấm Tạo ảnh để sinh ảnh tham khảo thực tế.");

    top3List.innerHTML = "";
    result.top3.forEach((item) => {
      const row = document.createElement("div");
      row.className = "top3-item";
      row.innerHTML = `
        <strong>${item.label} — ${item.meaning_vi}</strong>
        <span>${(item.confidence * 100).toFixed(2)}%</span>
      `;
      top3List.appendChild(row);
    });

    chatbotReply.innerHTML = markdownLite(result.chatbot_reply);
    resultBox.classList.remove("hidden");
  } catch (error) {
    chatbotReply.textContent = `Có lỗi xảy ra: ${error.message}`;
    resultBox.classList.remove("hidden");
  } finally {
    setLoading(false);
  }
}

async function generateReferenceImage() {
  if (!currentLabel) {
    alert("Bạn cần nhận diện hình vẽ trước khi tạo ảnh tham khảo.");
    return;
  }

  generateImageBtn.disabled = true;
  referenceStatus.textContent = "Đang tạo ảnh tham khảo...";
  referencePlaceholder.textContent = "Đang xử lý ảnh...";
  referenceImage.classList.add("hidden");

  try {
    const formData = new FormData();
    formData.append("label", currentLabel);
    const response = await fetch("/image/generate", {
      method: "POST",
      body: formData,
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.detail || "Không tạo được ảnh tham khảo.");

    referenceImage.src = result.image;
    referenceImage.classList.remove("hidden");
    referencePlaceholder.classList.add("hidden");
    referenceStatus.textContent =
      result.provider === "offline-pil-reference"
        ? "Đang dùng ảnh tham khảo offline. Thêm OPENAI_API_KEY để tạo ảnh photorealistic."
        : `Ảnh tạo bằng ${result.provider}.`;
  } catch (error) {
    resetReferenceImage("Không tạo được ảnh tham khảo.");
    referenceStatus.textContent = `Lỗi tạo ảnh: ${error.message}`;
  } finally {
    generateImageBtn.disabled = false;
  }
}

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseleave", stopDrawing);
canvas.addEventListener("touchstart", startDrawing, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
canvas.addEventListener("touchend", stopDrawing, { passive: false });

clearBtn.addEventListener("click", () => {
  resetCanvas();
  currentLabel = "";
  emptyState.classList.remove("hidden");
  resultBox.classList.add("hidden");
  resetReferenceImage();
  setQuickDrawGuesses();
  clearRecognizedPreview();
});

predictBtn.addEventListener("click", predictDrawing);
generateImageBtn.addEventListener("click", generateReferenceImage);
brushSize.addEventListener("input", () => {
  ctx.lineWidth = Number(brushSize.value);
});

resetCanvas();
clearRecognizedPreview();
renderTargetGuide();

startCameraBtn.addEventListener("click", () => startFaceCamera().catch(() => {}));
stopCameraBtn.addEventListener("click", stopFaceCamera);
showGuideToggle.addEventListener("change", renderTargetGuide);
enrollFaceBtn.addEventListener("click", () => sendFace("enroll"));
verifyFaceBtn.addEventListener("click", () => sendFace("verify"));
startHandDrawBtn.addEventListener("click", startHandDrawing);
stopHandDrawBtn.addEventListener("click", stopHandDrawing);
startQuickDrawBtn.addEventListener("click", startQuickDrawGame);
skipRoundBtn.addEventListener("click", skipQuickDrawRound);
clearAirDrawBtn.addEventListener("click", () => {
  resetCanvas();
  currentLabel = "";
  setQuickDrawGuesses();
  clearRecognizedPreview();
  setQuickDrawStatus(
    gameState.roundActive
      ? `Đã xóa nét. Tiếp tục vẽ ${gameState.target}.`
      : "Đã xóa nét vẽ."
  );
});
submitAirDrawBtn.addEventListener("click", () => predictForQuickDraw(true));
