const $ = (selector) => document.querySelector(selector);
const app = $("#app");

const state = {
  user: null,
  vocab: [],
  supportedVocab: [],
  gamePool: [],
  mode: "mouse",
  running: false,
  level: 1,
  score: 0,
  lives: 3,
  streak: 0,
  bestStreak: 0,
  correct: 0,
  attempts: 0,
  roundTime: 60,
  timeLeft: 60,
  currentTarget: null,
  prediction: null,
  judge: null,
  predictionBuffer: [],
  consecutiveCorrect: 0,
  hasDrawn: false,
  startedAt: 0,
  roundStartedAt: 0,
  timerId: null,
  realtimeId: null,
  predictInFlight: false,
  cameraStream: null,
  cameraTool: "hand",
  hands: null,
  handLoop: false,
  faceMesh: null,
  faceLoop: false,
  faceFilteredPoint: null,
  faceCenter: null,
  faceDrawingActive: false,
  faceStatus: null,
  faceAnalyzeInFlight: false,
  faceLastClear: 0,
  latestFaceMeshSketch: null,
  latestFaceMeshAt: 0,
  filteredPoint: null,
  currentStroke: null,
  strokes: [],
  lastPoint: null,
  lastPalmClear: 0,
  leaderboard: [],
  profile: null,
  pvpSocket: null,
  pvpRoom: "airdraw",
  pvpConnected: false,
  pvpEvents: [],
  retrainStatus: null,
  proStatus: null,
  strokeModelAvailable: false,
  lastSavedKey: "",
  imageModelReloaded: false,
  referenceImage: null,
  referenceImageInFlight: false,
  referenceImageRequestId: 0,
  referenceImageMessage: "Bấm Sinh hình thật sau khi vẽ xong để tạo ảnh.",
};

const CANVAS_W = 960;
const CANVAS_H = 540;
const SUCCESS_THRESHOLD_BASE = 0.68;
const REALTIME_INTERVAL_MS = 620;
const FACE_GAIN = 3.8;
const FACE_DRAW_MOUTH_THRESHOLD = 0.038;
const FACE_CLEAR_BLINK_THRESHOLD = 0.09;
const FACE_CENTER_DRIFT_ALPHA = 0.015;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function apiGet(url) {
  const res = await fetch(url, { credentials: "same-origin" });
  if (!res.ok) throw new Error(await extractError(res));
  return res.json();
}

async function apiPostForm(url, form) {
  const res = await fetch(url, {
    method: "POST",
    body: form,
    credentials: "same-origin",
  });
  if (!res.ok) throw new Error(await extractError(res));
  return res.json();
}

async function extractError(res) {
  try {
    const data = await res.json();
    const detail = data.detail || data.error;
    if (!detail) return `HTTP ${res.status}`;
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  } catch {
    return `HTTP ${res.status}`;
  }
}

async function boot() {
  renderLogin("Đang kiểm tra phiên đăng nhập...");
  try {
    const me = await apiGet("/auth/me");
    if (me.authenticated) {
      state.user = me.user;
      await loadGameData();
      renderGameShell();
      return;
    }
  } catch (err) {
    console.warn(err);
  }
  renderLogin();
}

function renderLogin(message = "") {
  stopAllLoops();
  stopCamera();
  app.innerHTML = `
    <main class="login-screen">
      <section class="login-card">
        <div class="login-logo">
          <div class="logo-mark">AD</div>
          <div>
            <p class="eyebrow">Final Boss Mode</p>
            <h1>AirDrawVocab</h1>
          </div>
        </div>
        <p class="login-subtitle">Đăng nhập để vào thẳng màn chơi. Không có dashboard phụ, không có giao diện người dùng thừa ở bên cạnh.</p>
        <form id="loginForm" class="login-form">
          <label>Tài khoản
            <input id="username" autocomplete="username" placeholder="VD: admin" required />
          </label>
          <label>Mật khẩu
            <input id="password" type="password" autocomplete="current-password" placeholder="Tối thiểu 6 ký tự" required />
          </label>
          <div class="login-actions">
            <button class="primary" id="loginBtn" type="submit">Đăng nhập</button>
            <button class="secondary" id="registerBtn" type="button">Đăng ký</button>
          </div>
          <p id="loginStatus" class="status">${escapeHtml(message || "Tạo tài khoản mới nếu chưa có.")}</p>
        </form>
      </section>
    </main>
  `;
  $("#loginForm").addEventListener("submit", (e) => {
    e.preventDefault();
    loginOrRegister("login");
  });
  $("#registerBtn").addEventListener("click", () =>
    loginOrRegister("register"),
  );
}

async function loginOrRegister(action) {
  const status = $("#loginStatus");
  const username = $("#username").value.trim();
  const password = $("#password").value;
  if (!username || !password) {
    status.textContent = "Nhập đủ tài khoản và mật khẩu.";
    status.className = "status error";
    return;
  }
  const form = new FormData();
  form.append("username", username);
  form.append("password", password);
  status.textContent =
    action === "login" ? "Đang đăng nhập..." : "Đang đăng ký...";
  status.className = "status";
  try {
    const data = await apiPostForm(`/auth/${action}`, form);
    state.user = data.user;
    await loadGameData();
    renderGameShell();
  } catch (err) {
    status.textContent = err.message;
    status.className = "status error";
  }
}

async function logout() {
  disconnectPvp();
  try {
    await apiPostForm("/auth/logout", new FormData());
  } catch {}
  state.user = null;
  renderLogin("Đã đăng xuất.");
}

async function loadGameData() {
  const data = await apiGet("/vocab");
  state.vocab = data.vocab || [];
  state.supportedVocab = state.vocab.filter(
    (item) => item.recognition_supported,
  );
  state.gamePool = state.supportedVocab.length
    ? state.supportedVocab
    : state.vocab;
  await refreshProfileAndLeaderboard();
}

async function refreshProfileAndLeaderboard() {
  try {
    state.profile = await apiGet("/game/profile");
  } catch {
    state.profile = null;
  }
  try {
    state.leaderboard = (await apiGet("/game/leaderboard")).leaderboard || [];
  } catch {
    state.leaderboard = [];
  }
}

function renderGameShell() {
  const recognitionNote =
    state.supportedVocab.length < state.vocab.length
      ? `<div class="warning-note">Model hiện tại nhận diện ${state.supportedVocab.length}/${state.vocab.length} từ. Game sẽ ưu tiên các từ model đang hỗ trợ; khi bạn thay model 40 lớp, game tự mở đủ 40 từ.</div>`
      : "";

  app.innerHTML = `
    <main class="game-shell">
      <header class="topbar">
        <div class="brand">
          <div class="logo-mark">AD</div>
          <div><div class="brand-title">AirDrawVocab</div><div class="brand-sub">Realtime AI Drawing Game</div></div>
        </div>
        <div class="hud">
          <div class="hud-card"><span>Draw</span><strong id="hudTarget">---</strong></div>
          <div class="hud-card"><span>Score</span><strong id="hudScore">0</strong></div>
          <div class="hud-card"><span>Time</span><strong id="hudTime">60</strong></div>
          <div class="hud-card"><span>Level</span><strong id="hudLevel">1/${state.gamePool.length || 40}</strong></div>
          <div class="hud-card"><span>Lives</span><strong id="hudLives">♥♥♥</strong></div>
          <div class="hud-card"><span>Streak</span><strong id="hudStreak">0</strong></div>
        </div>
        <div class="user-actions">
          <span class="user-pill">${escapeHtml(state.user?.username || "guest")}</span>
          <button id="logoutBtn" class="secondary">Đăng xuất</button>
        </div>
      </header>

      <section class="game-layout">
        <section class="stage-card">
          <div class="stage-header">
            <div><h2>Final Boss Game Arena</h2></div>
            <div class="mode-switch">
              <button id="mouseModeBtn" class="secondary active">Vẽ chuột</button>
              <button id="cameraModeBtn" class="secondary">Vẽ tay camera</button>
            </div>
            <div class="camera-tool-switch" id="cameraToolSwitch">
              <span>Bút camera</span>
              <button id="handToolBtn" class="secondary active" type="button">Bút tay</button>
              <button id="faceToolBtn" class="secondary" type="button">Bút mặt/mũi</button>
              <button id="faceSnapBtn" class="secondary" type="button" disabled>Quét nét mặt</button>
            </div>
          </div>

          <div id="gameStage" class="game-stage mouse-mode">
            <video id="cameraVideo" autoplay muted playsinline></video>
            <canvas id="drawCanvas" width="${CANVAS_W}" height="${CANVAS_H}"></canvas>
            <canvas id="handCanvas" width="${CANVAS_W}" height="${CANVAS_H}"></canvas>
            <canvas id="faceCanvas" width="${CANVAS_W}" height="${CANVAS_H}"></canvas>
            <div class="stage-overlay">
              <div class="target-chip">Mục tiêu: <strong id="targetChip">---</strong></div>
              <div class="ai-chip">AI thấy: <strong id="aiChip">---</strong></div>
              <div class="tool-chip">Bút: <strong id="toolChip">tay/ngón trỏ</strong></div>
              <div class="face-chip">Mặt: <strong id="faceChip">tắt</strong></div>
            </div>
            <div id="correctFlash" class="correct-flash">Correct +Score</div>
          </div>

          <div class="controls">
            <div class="control-row">
              <button id="startBtn" class="primary">Bắt đầu</button>
              <button id="clearBtn" class="secondary">Xóa nét</button>
              <button id="skipBtn" class="secondary" disabled>Bỏ qua</button>
              <button id="recognizeBtn" class="primary" type="button" disabled>Nhận diện</button>
              <button id="generateImageBtn" class="secondary" type="button" disabled>Sinh hình thật</button>
              <button id="saveBtn" class="secondary" disabled>Lưu mẫu train</button>
              <span class="spacer"></span>
            </div>
            <div class="progress"><span id="confidenceProgress"></span></div>
            <p id="gameStatus" class="status">Bấm Bắt đầu để chơi. Vẽ xong rồi bấm Nhận diện hoặc Sinh hình thật; AI không tự đoán và không tự sinh ảnh trong lúc bạn vẽ.</p>
          </div>
        </section>

        <aside class="side-panel">
          <section class="side-card">
            <div class="side-header"><h3>Từ cần vẽ</h3><span class="badge">40 vocab</span></div>
            <h2 id="targetWord" class="target-word">---</h2>
            <div id="wordMeta" class="word-meta">${recognitionNote || "Chưa bắt đầu game."}</div>
            <button id="speakBtn" class="ghost">Đọc từ / ví dụ</button>
          </section>

          <section class="side-card">
            <div class="side-header"><h3>Nhận diện AI</h3><span class="badge">Manual</span></div>
            <div id="predictionList" class="prediction-list">AI chưa có dữ liệu. Vẽ xong bấm Nhận diện.</div>
          </section>

          <section class="side-card">
            <div class="side-header"><h3>Hình thật sau khi vẽ</h3><span id="realImageBadge" class="badge warn">Chờ bấm</span></div>
            <div id="realImageBox" class="real-image-box empty">Vẽ xong rồi bấm Sinh hình thật để tạo ảnh. Hệ thống sẽ không tự sinh ảnh trong lúc bạn vẽ.</div>
            <div class="pvp-controls two storage-controls">
              <button id="realImageStorageBtn" class="secondary" type="button">Storage hình thật</button>
              <button id="panelStorageBtn" class="secondary" type="button">Storage tất cả</button>
            </div>
            <div id="panelStorageBox" class="mini-list"></div>
          </section>

          <section class="side-card">
            <div class="side-header"><h3>AI Judge Mode</h3><span id="gradeBadge" class="badge warn">---</span></div>
            <div class="judge-grid">
              <div class="judge-cell"><span>Shape</span><strong id="shapeScore">0</strong></div>
              <div class="judge-cell"><span>Clarity</span><strong id="clarityScore">0</strong></div>
              <div class="judge-cell"><span>Stroke</span><strong id="strokeScore">0</strong></div>
              <div class="judge-cell"><span>Speed</span><strong id="speedScore">0</strong></div>
            </div>
            <p id="judgeFeedback" class="feedback">AI sẽ đánh giá khi bạn bấm Nhận diện.</p>
          </section>

          <section class="side-card">
            <div class="side-header"><h3>Skill Profile</h3><span class="badge">Adaptive</span></div>
            <div id="profileBox" class="profile-grid"></div>
            <div id="weaknessBox" class="mini-list"></div>
          </section>

          <section class="side-card">
            <div class="side-header"><h3>Leaderboard</h3><span class="badge">SQLite</span></div>
            <div id="leaderboardBox" class="mini-list"></div>
          </section>

          <section class="side-card">
            <div class="side-header"><h3>PvP WebSocket</h3><span id="pvpBadge" class="badge warn">OFF</span></div>
            <div class="pvp-controls">
              <input id="pvpRoomInput" value="${escapeHtml(state.pvpRoom)}" placeholder="room name" />
              <button id="pvpBtn" class="secondary" type="button">Join</button>
            </div>
            <div id="pvpBox" class="mini-list"><div class="mini-row"><span>Chưa vào phòng</span><small>Solo mode</small></div></div>
          </section>

          <section class="side-card">
            <div class="side-header"><h3>Self-improving Loop</h3><span class="badge">Train</span></div>
            <div class="pvp-controls four">
              <button id="exportDatasetBtn" class="secondary" type="button">Export data</button>
              <button id="trainStrokeBtn" class="secondary" type="button">Train stroke</button>
              <button id="trainImageBtn" class="secondary" type="button">Train image</button>
              <button id="storageBtn" class="secondary" type="button">Storage</button>
            </div>
            <div id="retrainBox" class="mini-list"><div class="mini-row"><span>Idle</span><small>Local/Colab pipeline</small></div></div>
          </section>

          <section class="side-card">
            <div class="side-header"><h3>Production AI Ops</h3><span class="badge">MLOps</span></div>
            <div class="pvp-controls three">
              <button id="buildBenchmarkBtn" class="secondary" type="button">Build benchmark</button>
              <button id="evalReleaseBtn" class="secondary" type="button">Evaluate</button>
              <button id="promoteDryRunBtn" class="secondary" type="button">Promote check</button>
            </div>
            <div id="proOpsBox" class="mini-list"><div class="mini-row"><span>Đang tải</span><small>benchmark / eval / promotion</small></div></div>
          </section>
        </aside>
      </section>
    </main>
  `;

  wireGameEvents();
  setupCanvas();
  updateHud();
  updateRecognizeButton();
  updateReferenceImagePanel();
  renderProfile();
  renderLeaderboard();
  renderPvp();
  refreshRetrainStatus();
  refreshProStatus();
}

function wireGameEvents() {
  $("#logoutBtn").addEventListener("click", logout);
  $("#mouseModeBtn").addEventListener("click", () => setMode("mouse"));
  $("#cameraModeBtn").addEventListener("click", () => setMode("camera"));
  $("#handToolBtn")?.addEventListener("click", () => setCameraTool("hand"));
  $("#faceToolBtn")?.addEventListener("click", () => setCameraTool("face"));
  $("#faceSnapBtn")?.addEventListener("click", () => scanFaceSketch(false));
  $("#startBtn").addEventListener("click", () =>
    state.running ? endGame("Bạn đã kết thúc lượt chơi.") : startGame(),
  );
  $("#clearBtn").addEventListener("click", clearDrawing);
  $("#skipBtn").addEventListener("click", skipRound);
  $("#recognizeBtn").addEventListener("click", manualRecognizeDrawing);
  $("#generateImageBtn")?.addEventListener("click", manualGenerateReferenceImage);
  $("#realImageStorageBtn")?.addEventListener("click", () => showPanelStorage("real_image_after_draw"));
  $("#panelStorageBtn")?.addEventListener("click", () => showPanelStorage(""));
  $("#saveBtn").addEventListener("click", () => saveStrokeSample(Boolean(state.prediction?.is_correct), true));
  $("#speakBtn").addEventListener("click", speakCurrentWord);
  $("#pvpBtn").addEventListener("click", togglePvp);
  $("#exportDatasetBtn").addEventListener("click", exportDataset);
  $("#trainStrokeBtn").addEventListener("click", () => startRetrain("stroke"));
  $("#trainImageBtn").addEventListener("click", () => startRetrain("image"));
  $("#storageBtn")?.addEventListener("click", showSelfLoopStorage);
  $("#buildBenchmarkBtn")?.addEventListener("click", buildBenchmark);
  $("#evalReleaseBtn")?.addEventListener("click", evaluateRelease);
  $("#promoteDryRunBtn")?.addEventListener("click", promoteDryRun);
  updateCameraToolUI();
}


function currentInputMode() {
  return state.mode === "camera" ? `camera-${state.cameraTool}` : state.mode;
}

function cameraInstruction() {
  return state.cameraTool === "face"
    ? "Bút mặt/mũi: há miệng nhẹ để vẽ bằng mũi, ngậm miệng để nhấc bút, chớp cả hai mắt để xóa. Nút Quét nét mặt sẽ chuyển khuôn mặt hiện tại thành stroke."
    : "Bút tay: giơ ngón trỏ để vẽ, xòe bàn tay để xóa nét.";
}

function cameraFaceLabel() {
  if (state.cameraTool !== "face") return "chờ";
  if (state.faceAnalyzeInFlight) return "đang quét mặt";
  if (state.faceDrawingActive) return "đang vẽ bằng mũi";
  if (state.faceStatus?.faceDetected) return state.faceStatus.ready ? state.faceStatus.status || "mặt sẵn sàng" : state.faceStatus.status || "chỉnh mặt";
  return "chưa thấy mặt";
}

function updateCameraToolUI() {
  const handBtn = $("#handToolBtn");
  const faceBtn = $("#faceToolBtn");
  const snapBtn = $("#faceSnapBtn");
  const stage = $("#gameStage");
  handBtn?.classList.toggle("active", state.cameraTool === "hand");
  faceBtn?.classList.toggle("active", state.cameraTool === "face");
  stage?.classList.toggle("face-tracking", state.cameraTool === "face");
  stage?.classList.toggle("hand-tracking", state.cameraTool === "hand");
  if (snapBtn) {
    snapBtn.disabled = state.mode !== "camera" || state.faceAnalyzeInFlight;
    snapBtn.textContent = state.faceAnalyzeInFlight ? "Đang quét..." : "Quét nét mặt";
  }
  const toolChip = $("#toolChip");
  if (toolChip) toolChip.textContent = state.cameraTool === "face" ? "mặt/mũi" : "tay/ngón trỏ";
  const faceChip = $("#faceChip");
  if (faceChip) faceChip.textContent = state.mode === "camera" ? cameraFaceLabel() : "tắt";
}

async function setCameraTool(tool) {
  state.cameraTool = tool === "face" ? "face" : "hand";
  state.handLoop = false;
  state.faceLoop = false;
  state.filteredPoint = null;
  if (typeof resetHandEuro === "function") resetHandEuro();
  state.faceFilteredPoint = null;
  if (typeof resetFaceEuro === "function") resetFaceEuro();
  state.faceCenter = null;
  state.faceDrawingActive = false;
  state.penLiftFrames = 0;
  $("#handCanvas")?.getContext("2d")?.clearRect(0, 0, CANVAS_W, CANVAS_H);
  $("#faceCanvas")?.getContext("2d")?.clearRect(0, 0, CANVAS_W, CANVAS_H);
  updateCameraToolUI();
  if (state.mode === "camera") {
    const ok = await startCamera();
    if (ok) setStatus(cameraInstruction(), "ok");
  } else {
    setStatus(state.cameraTool === "face" ? "Đã chọn bút mặt/mũi. Chuyển sang chế độ camera để dùng." : "Đã chọn bút tay/ngón trỏ.");
  }
}

function setupCanvas() {
  const canvas = $("#drawCanvas");
  const ctx = canvas.getContext("2d");
  ctx.lineWidth = 13;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#050505";

  canvas.addEventListener("pointerdown", (event) => {
    if (state.mode !== "mouse") return;
    event.preventDefault();
    beginStroke(pointerToCanvas(event));
  });
  canvas.addEventListener("pointermove", (event) => {
    if (state.mode !== "mouse" || !state.currentStroke) return;
    event.preventDefault();
    extendStroke(pointerToCanvas(event));
  });
  window.addEventListener("pointerup", () => endStroke());
}

function pointerToCanvas(event) {
  const rect = $("#drawCanvas").getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * CANVAS_W,
    y: ((event.clientY - rect.top) / rect.height) * CANVAS_H,
    t: performance.now(),
  };
}

function beginStroke(point) {
  state.currentStroke = [point];
  state.lastPoint = point;
  state.lastMid = point; // điểm giữa khởi đầu để vẽ đường cong LIỀN MẠCH
  state.hasDrawn = true;
  updateRecognizeButton();
}

function extendStroke(point) {
  if (!state.lastPoint) {
    beginStroke(point);
    return;
  }
  const dx = point.x - state.lastPoint.x;
  const dy = point.y - state.lastPoint.y;
  const dist = Math.hypot(dx, dy);
  // CHỐNG "KÉO NÉT": nếu đầu ngón nhảy quá xa giữa 2 khung (thường do nhấc tay
  // rồi đặt sang chỗ khác), KHÔNG nối liền mà KẾT THÚC nét cũ và BẮT ĐẦU nét mới
  // -> mỗi nét tách bạch, không bị đường thẳng nối các nét như trước.
  const MAX_JUMP = 150;
  if (dist > MAX_JUMP) {
    endStroke();
    beginStroke(point);
    return;
  }
  // Nội suy: nếu khoảng cách vừa phải nhưng hơi lớn (tay di nhanh), chèn điểm
  // trung gian để nét trong CÙNG một stroke vẫn liền mạch.
  const STEP = 18;
  if (dist > STEP * 1.5) {
    const steps = Math.min(20, Math.floor(dist / STEP));
    for (let i = 1; i < steps; i++) {
      const t = i / steps;
      const mid = {
        x: state.lastPoint.x + dx * t,
        y: state.lastPoint.y + dy * t,
        t: state.lastPoint.t + (point.t - state.lastPoint.t) * t,
      };
      drawSegment(state.lastPoint, mid);
      state.currentStroke.push(mid);
      state.lastPoint = mid;
    }
  }
  drawSegment(state.lastPoint, point);
  state.currentStroke.push(point);
  state.lastPoint = point;
  state.hasDrawn = true;
  updateRecognizeButton();
}

function endStroke() {
  if (state.currentStroke && state.currentStroke.length > 0) {
    state.strokes.push(state.currentStroke);
  }
  state.currentStroke = null;
  state.lastPoint = null;
  updateRecognizeButton();
  if (state.hasDrawn && !state.referenceImage && !state.referenceImageInFlight) {
    state.referenceImageMessage = "Đã có nét vẽ. Bấm Sinh hình thật để tạo ảnh tham khảo.";
    updateReferenceImagePanel();
  }
}

function drawSegment(a, b) {
  const ctx = $("#drawCanvas").getContext("2d");
  // Kỹ thuật "quadratic qua điểm giữa": mỗi đoạn bắt đầu ĐÚNG nơi đoạn trước kết
  // thúc (state.lastMid) nên nét luôn liền mạch, không hở như trước.
  const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  const start = state.lastMid || a;
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.quadraticCurveTo(a.x, a.y, mid.x, mid.y);
  ctx.stroke();
  state.lastMid = mid;
}


function clearFaceOverlay() {
  const face = $("#faceCanvas");
  if (face) face.getContext("2d").clearRect(0, 0, CANVAS_W, CANVAS_H);
}

function captureVideoFrameBlob(video, targetWidth = 640) {
  return new Promise((resolve, reject) => {
    if (!video || video.readyState < 2) {
      reject(new Error("Camera chưa có khung hình dùng được."));
      return;
    }
    const srcW = video.videoWidth || CANVAS_W;
    const srcH = video.videoHeight || CANVAS_H;
    const scale = targetWidth / Math.max(1, srcW);
    const temp = document.createElement("canvas");
    temp.width = targetWidth;
    temp.height = Math.max(1, Math.round(srcH * scale));
    const ctx = temp.getContext("2d");
    ctx.drawImage(video, 0, 0, temp.width, temp.height);
    temp.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Không chụp được khung hình camera."));
    }, "image/jpeg", 0.78);
  });
}

function normalizeStrokePoint(point, index, source) {
  const x = Math.max(0, Math.min(CANVAS_W, Number(point?.x ?? 0)));
  const y = Math.max(0, Math.min(CANVAS_H, Number(point?.y ?? 0)));
  return {
    x,
    y,
    t: Number(point?.t ?? performance.now() + index * 8),
    source: point?.source || source,
  };
}

function drawDetectedFaceOverlay(data) {
  const canvas = $("#faceCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
  if (!data?.face_detected) return;
  ctx.save();
  ctx.lineWidth = 2.5;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(85, 230, 165, 0.78)";
  for (const stroke of data.strokes || []) {
    if (!Array.isArray(stroke) || stroke.length < 2) continue;
    ctx.beginPath();
    stroke.forEach((point, idx) => {
      const p = normalizeStrokePoint(point, idx, "camera-face-sketch");
      if (idx === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
  }
  const box = data.face_bbox || data.bbox;
  if (box) {
    ctx.strokeStyle = "rgba(87, 167, 255, 0.92)";
    ctx.lineWidth = 3;
    ctx.strokeRect(Number(box.x || 0), Number(box.y || 0), Number(box.width || 0), Number(box.height || 0));
  }
  ctx.restore();
}


function chainToStroke(landmarks, indices, source, tOffset = 0) {
  const stroke = [];
  indices.forEach((idx, pointIndex) => {
    const p = landmarks[idx];
    if (!p) return;
    stroke.push({
      x: Math.max(0, Math.min(CANVAS_W, (1 - p.x) * CANVAS_W)),
      y: Math.max(0, Math.min(CANVAS_H, p.y * CANVAS_H)),
      t: performance.now() + tOffset + pointIndex * 7,
      source,
    });
  });
  return stroke;
}

function makeFaceMeshSketchData(results) {
  const landmarks = results?.multiFaceLandmarks?.[0];
  if (!landmarks) {
    return {
      ok: true,
      face_detected: false,
      detector: "browser-facemesh",
      message: "Chưa thấy khuôn mặt rõ trong camera.",
      strokes: [],
      stroke_count: 0,
      point_count: 0,
      quality: 0,
    };
  }

  const chains = [
    { source: "face-mesh-oval", idx: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10] },
    { source: "face-mesh-left-brow", idx: [70, 63, 105, 66, 107] },
    { source: "face-mesh-right-brow", idx: [336, 296, 334, 293, 300] },
    { source: "face-mesh-left-eye", idx: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33] },
    { source: "face-mesh-right-eye", idx: [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263] },
    { source: "face-mesh-nose", idx: [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 2] },
    { source: "face-mesh-mouth", idx: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61] },
    { source: "face-mesh-lip-inner", idx: [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78] },
  ];
  const strokes = chains
    .map((chain, i) => chainToStroke(landmarks, chain.idx, chain.source, i * 180))
    .filter((stroke) => stroke.length >= 2);
  const xs = landmarks.map((p) => (1 - p.x) * CANVAS_W);
  const ys = landmarks.map((p) => p.y * CANVAS_H);
  const minX = Math.max(0, Math.min(...xs));
  const maxX = Math.min(CANVAS_W, Math.max(...xs));
  const minY = Math.max(0, Math.min(...ys));
  const maxY = Math.min(CANVAS_H, Math.max(...ys));
  const pointCount = strokes.reduce((sum, stroke) => sum + stroke.length, 0);
  return {
    ok: true,
    face_detected: true,
    detector: "browser-facemesh-landmarks",
    message: "Đã nhận diện outline, mắt, mũi, miệng bằng FaceMesh.",
    face_bbox: { x: minX, y: minY, width: maxX - minX, height: maxY - minY },
    strokes,
    semantic_strokes: strokes,
    edge_strokes: [],
    stroke_count: strokes.length,
    point_count: pointCount,
    quality: Math.max(40, Math.min(100, Math.round(pointCount * 0.55))),
  };
}

function freshFaceMeshSketch(maxAgeMs = 1800) {
  if (!state.latestFaceMeshSketch?.face_detected) return null;
  if (performance.now() - state.latestFaceMeshAt > maxAgeMs) return null;
  if (!(state.latestFaceMeshSketch.strokes || []).length) return null;
  return state.latestFaceMeshSketch;
}

function handleFaceMeshUnifiedResults(results) {
  const sketch = makeFaceMeshSketchData(results);
  state.latestFaceMeshSketch = sketch;
  state.latestFaceMeshAt = performance.now();
  if (state.cameraTool === "face") {
    onFaceResults(results);
  }
}

function hasFaceStrokesInDrawing() {
  return getAllCurrentStrokes().some((stroke) =>
    Array.isArray(stroke) && stroke.some((point) => {
      const source = String(point?.source || "");
      return source.startsWith("face-") || source.includes("camera-face");
    }),
  );
}

function drawStrokeListToCanvas(strokes, source = "camera-face-sketch") {
  const canvas = $("#drawCanvas");
  if (!canvas) return 0;
  const ctx = canvas.getContext("2d");
  let added = 0;
  ctx.save();
  ctx.lineWidth = 11;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#050505";
  for (const stroke of strokes || []) {
    if (!Array.isArray(stroke) || stroke.length < 2) continue;
    const clean = stroke
      .map((point, index) => normalizeStrokePoint(point, index, source))
      .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
    if (clean.length < 2) continue;
    ctx.beginPath();
    clean.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();
    state.strokes.push(clean);
    added += 1;
  }
  ctx.restore();
  if (added > 0) {
    state.hasDrawn = true;
    updateRecognizeButton();
    if (!state.referenceImage && !state.referenceImageInFlight) {
      state.referenceImageMessage = "Đã có nét vẽ. Bấm Sinh hình thật để tạo ảnh tham khảo.";
      updateReferenceImagePanel();
    }
  }
  return added;
}

async function scanFaceSketch(auto = false) {
  if (state.faceAnalyzeInFlight) return;
  if (state.mode !== "camera") {
    await setMode("camera");
    if (state.mode !== "camera") return;
  }
  if (state.cameraTool !== "face") {
    await setCameraTool("face");
  }
  const video = $("#cameraVideo");
  if (!video || !state.cameraStream) {
    const ok = await startCamera();
    if (!ok) return;
  }
  state.faceAnalyzeInFlight = true;
  state.faceStatus = { faceDetected: true, ready: false, status: "đang quét" };
  updateCameraToolUI();
  if (!auto) setStatus("Đang quét khuôn mặt trong khung camera để tạo nét vẽ...");
  try {
    let data = freshFaceMeshSketch();
    if (!data && state.faceMesh) {
      // Chờ FaceMesh cập nhật thêm 1-2 frame trước khi dùng fallback backend.
      await waitNextFrame();
      data = freshFaceMeshSketch(2400);
    }
    if (!data) {
      const blob = await captureVideoFrameBlob($("#cameraVideo"), 640);
      const form = new FormData();
      form.append("file", blob, "camera-face-frame.jpg");
      form.append("canvas_width", String(CANVAS_W));
      form.append("canvas_height", String(CANVAS_H));
      form.append("mirror", "1");
      form.append("preview", "0");
      data = await apiPostForm("/camera/face-strokes", form);
    }
    drawDetectedFaceOverlay(data);
    if (!data?.face_detected || !(data.strokes || []).length) {
      const message = data?.message || "Chưa nhận diện được nét mặt. Hãy nhìn thẳng camera và tăng ánh sáng.";
      state.faceStatus = { faceDetected: false, ready: false, status: message };
      updateCameraToolUI();
      if (!auto) setStatus(message, "error");
      return;
    }
    const source = String(data.detector || "").includes("facemesh") ? "face-mesh-sketch" : "camera-face-sketch";
    const added = drawStrokeListToCanvas(data.strokes, source);
    state.faceStatus = {
      faceDetected: true,
      ready: true,
      status: `${added} nét mặt`,
    };
    updateCameraToolUI();
    if (added > 0) {
      setStatus(`Đã thêm ${added} nét khuôn mặt vào canvas vẽ (${escapeHtml(data.detector || "camera")}).`, "ok");
      updateRecognizeButton();
      if (!state.referenceImage && !state.referenceImageInFlight) {
        state.referenceImageMessage = "Đã có nét vẽ. Bấm Sinh hình thật để tạo ảnh tham khảo.";
        updateReferenceImagePanel();
      }
    } else if (!auto) {
      setStatus("Đã thấy mặt nhưng nét sinh ra quá nhỏ/ít. Hãy đưa mặt gần camera hơn.", "error");
    }
  } catch (err) {
    clearFaceOverlay();
    state.faceStatus = { faceDetected: false, ready: false, status: err.message };
    updateCameraToolUI();
    if (!auto) setStatus(err.message, "error");
  } finally {
    state.faceAnalyzeInFlight = false;
    updateCameraToolUI();
  }
}


function clearDrawing() {
  const draw = $("#drawCanvas");
  draw.getContext("2d").clearRect(0, 0, CANVAS_W, CANVAS_H);
  const hand = $("#handCanvas");
  hand.getContext("2d").clearRect(0, 0, CANVAS_W, CANVAS_H);
  clearFaceOverlay();
  state.hasDrawn = false;
  state.currentStroke = null;
  state.strokes = [];
  state.lastPoint = null;
  state.lastMid = null;
  state.penLiftFrames = 0;
  state.filteredPoint = null;
  if (typeof resetHandEuro === "function") resetHandEuro();
  state.predictionBuffer = [];
  state.consecutiveCorrect = 0;
  state.prediction = null;
  state.judge = null;
  resetReferenceImage("Vẽ xong rồi bấm Sinh hình thật để tạo ảnh. Hệ thống sẽ không tự sinh ảnh trong lúc bạn vẽ.");
  updatePredictionPanel(null);
  updateJudge(null);
  $("#aiChip").textContent = "---";
  $("#confidenceProgress").style.width = "0%";
  updateRecognizeButton();
}

async function setMode(mode) {
  state.mode = mode;
  const stage = $("#gameStage");
  stage.classList.toggle("camera-mode", mode === "camera");
  stage.classList.toggle("mouse-mode", mode === "mouse");
  $("#mouseModeBtn").classList.toggle("active", mode === "mouse");
  $("#cameraModeBtn").classList.toggle("active", mode === "camera");
  updateCameraToolUI();
  if (mode === "camera") {
    const ok = await startCamera();
    if (ok) {
      setStatus(cameraInstruction(), "ok");
    } else {
      state.mode = "mouse";
      stage.classList.remove("camera-mode");
      stage.classList.add("mouse-mode");
      $("#mouseModeBtn").classList.add("active");
      $("#cameraModeBtn").classList.remove("active");
      updateCameraToolUI();
    }
  } else {
    clearFaceOverlay();
    stopCamera();
    setStatus("Chế độ vẽ chuột: vẽ trực tiếp trên canvas trắng.");
  }
}


function shuffle(items) {
  const arr = [...items];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

async function startGame() {
  if (!state.gamePool.length) {
    setStatus("Chưa tải được từ vựng.", "error");
    return;
  }
  state.running = true;
  state.level = 1;
  state.score = 0;
  state.lives = 3;
  state.streak = 0;
  state.bestStreak = 0;
  state.correct = 0;
  state.attempts = 0;
  state.startedAt = Date.now();
  state.gamePool = shuffle(
    state.supportedVocab.length ? state.supportedVocab : state.vocab,
  );
  $("#startBtn").textContent = "Kết thúc";
  $("#skipBtn").disabled = false;
  $("#saveBtn").disabled = false;
  nextRound();
  startGameTimer();
  startRealtimeAI();
  if (state.mode === "camera") await startCamera();
  updateRecognizeButton();
  setStatus(
    "Final Boss Mode đang chạy: hãy vẽ xong rồi bấm Nhận diện hoặc Sinh hình thật. AI không tự đoán và không tự sinh ảnh trong lúc bạn vẽ.",
    "ok",
  );
}

function nextRound() {
  if (!state.running) return;
  if (state.level > state.gamePool.length) {
    endGame("Hoàn thành toàn bộ vòng chơi.");
    return;
  }
  clearDrawing();
  state.currentTarget =
    state.gamePool[(state.level - 1) % state.gamePool.length];
  state.roundTime = Math.max(35, 60 - Math.min(state.streak, 10) * 2);
  state.timeLeft = state.roundTime;
  state.roundStartedAt = Date.now();
  state.lastSavedKey = "";
  resetReferenceImage("Vẽ xong rồi bấm Sinh hình thật để tạo ảnh. Hệ thống sẽ không tự sinh ảnh trong lúc bạn vẽ.");
  updateTargetPanel();
  updateHud();
}

function startGameTimer() {
  clearInterval(state.timerId);
  state.timerId = setInterval(() => {
    if (!state.running) return;
    state.timeLeft -= 1;
    if (state.timeLeft <= 0) {
      failRound("Hết giờ.");
    }
    updateHud();
  }, 1000);
}

function startRealtimeAI() {
  // Theo yêu cầu mới: không nhận diện tự động trong lúc người dùng đang vẽ.
  // Hàm vẫn được giữ để các luồng cũ gọi an toàn, nhưng không tạo interval nữa.
  clearInterval(state.realtimeId);
  state.realtimeId = null;
}

async function manualRecognizeDrawing() {
  if (!state.running || !state.currentTarget) {
    setStatus("Bấm Bắt đầu trước khi nhận diện.", "error");
    return;
  }
  if (!state.hasDrawn) {
    setStatus("Bạn hãy vẽ xong rồi mới bấm Nhận diện.", "error");
    return;
  }
  if (state.predictInFlight) return;
  setStatus("Đang nhận diện nét vẽ...", "ok");
  await realtimePredict({ manual: true });
}

async function realtimePredict({ manual = true } = {}) {
  state.predictInFlight = true;
  updateRecognizeButton();
  try {
    const blob = await captureDrawingBlob();
    const form = new FormData();
    form.append("file", blob, "drawing.png");
    form.append("target", state.currentTarget.label);
    form.append("source", currentInputMode());
    form.append("stroke_count", String(countStrokePoints()));
    form.append("elapsed_ms", String(Date.now() - state.roundStartedAt));
    form.append("strokes_json", JSON.stringify(getAllCurrentStrokes()));
    let data = await apiPostForm("/predict_godmode", form);
    const strokeData = await tryStrokePrediction();
    if (
      strokeData?.available &&
      strokeData.confidence > (data.confidence || 0)
    ) {
      data = {
        ...data,
        label: strokeData.label,
        confidence: strokeData.confidence,
        confidence_percent: Math.round(strokeData.confidence * 10000) / 100,
        top5: strokeData.top5,
        is_correct: strokeData.is_correct,
        ai_source: "stroke-sequence",
        judge: {
          ...(data.judge || {}),
          predicted: strokeData.label,
          correct: strokeData.is_correct,
          clarity_score: Math.round(strokeData.confidence * 100),
          feedback: strokeData.is_correct
            ? "Stroke model nhận đúng theo chuỗi nét bạn vừa vẽ."
            : `Stroke model đoán '${strokeData.label}'. Hãy lưu mẫu train để model học thêm kiểu vẽ của bạn.`,
        },
      };
    } else {
      data.ai_source = data.ai_source || data.source || "image-cnn";
    }
    if (hasFaceStrokesInDrawing()) {
      data.ai_source = `${data.ai_source}+face-strokes`;
    }
    state.prediction = data;
    state.judge = data.judge;
    updatePredictionPanel(data);
    updateJudge(data.judge);
    broadcastPvp({
      type: "prediction",
      label: data.label,
      confidence: data.confidence,
      target: state.currentTarget?.label,
      score: state.score,
    });
    handleRecognitionDecision(data, { manual });
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    state.predictInFlight = false;
    updateRecognizeButton();
  }
}

function captureDrawingBlob() {
  return new Promise((resolve) => {
    const src = $("#drawCanvas");
    const temp = document.createElement("canvas");
    temp.width = CANVAS_W;
    temp.height = CANVAS_H;
    const ctx = temp.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.drawImage(src, 0, 0);
    temp.toBlob((blob) => resolve(blob), "image/png");
  });
}

function handleRecognitionDecision(data, { manual = true } = {}) {
  const conf = Number(data.confidence || 0);
  const target = state.currentTarget?.label;
  $("#aiChip").textContent = `${data.label} ${(conf * 100).toFixed(0)}%`;
  $("#confidenceProgress").style.width = `${Math.round(conf * 100)}%`;
  state.predictionBuffer.push({ label: data.label, confidence: conf });
  if (state.predictionBuffer.length > 6) state.predictionBuffer.shift();
  const threshold = Math.max(
    0.58,
    SUCCESS_THRESHOLD_BASE - Math.min(state.streak, 8) * 0.01,
  );
  const isCorrect = data.label === target && conf >= threshold;

  // Không tự sinh ảnh sau khi nhận diện. Người dùng bấm riêng nút Sinh hình thật
  // nếu muốn xem ảnh theo nhãn AI vừa đoán hoặc theo mục tiêu hiện tại.

  if (manual) {
    state.consecutiveCorrect = isCorrect ? 1 : 0;
    if (isCorrect) {
      setStatus(`Nhận diện đúng '${target}' (${Math.round(conf * 100)}%). Chuyển sang từ tiếp theo...`, "ok");
      setTimeout(() => passRound(conf), 650);
    } else {
      setStatus(
        `AI đoán '${data.label}' (${Math.round(conf * 100)}%), mục tiêu là '${target}'. Bạn có thể sửa nét rồi bấm Nhận diện lại.`,
        "error",
      );
    }
    return;
  }

  if (isCorrect) {
    state.consecutiveCorrect += 1;
  } else {
    state.consecutiveCorrect = Math.max(0, state.consecutiveCorrect - 1);
  }
  if (state.consecutiveCorrect >= 2) passRound(conf);
}

function passRound(confidence) {
  if (!state.running) return;
  state.attempts += 1;
  state.correct += 1;
  const timeBonus = Math.max(0, state.timeLeft);
  const streakBonus = state.streak * 3;
  const confidenceBonus = Math.round(confidence * 20);
  const gained = 20 + timeBonus + streakBonus + confidenceBonus;
  state.score += gained;
  state.streak += 1;
  state.bestStreak = Math.max(state.bestStreak, state.streak);
  state.consecutiveCorrect = 0;
  showCorrectFlash(`Correct +${gained}`);
  broadcastPvp({
    type: "score",
    score: state.score,
    target: state.currentTarget?.label,
    message: `Correct +${gained}`,
  });
  saveStrokeSample(true);
  setTimeout(() => {
    state.level += 1;
    nextRound();
  }, 850);
  updateHud();
}

function failRound(reason) {
  state.attempts += 1;
  state.streak = 0;
  state.lives -= 1;
  saveStrokeSample(false);
  if (state.lives <= 0) {
    endGame(`${reason} Hết mạng.`);
  } else {
    setStatus(`${reason} Mất 1 mạng, chuyển từ tiếp theo.`, "error");
    state.level += 1;
    nextRound();
  }
}

function skipRound() {
  if (!state.running) return;
  failRound("Bạn đã bỏ qua.");
}

async function endGame(message) {
  if (!state.running) return;
  state.running = false;
  stopAllLoops();
  $("#startBtn").textContent = "Bắt đầu";
  $("#skipBtn").disabled = true;
  updateRecognizeButton();
  const duration = Math.round((Date.now() - state.startedAt) / 1000);
  const accuracy = state.attempts ? (state.correct / state.attempts) * 100 : 0;
  const form = new FormData();
  form.append("score", String(state.score));
  form.append("level", String(state.level));
  form.append("streak", String(state.bestStreak));
  form.append("accuracy", String(accuracy.toFixed(1)));
  form.append("duration_seconds", String(duration));
  form.append("mode", currentInputMode());
  try {
    await apiPostForm("/game/session", form);
  } catch (err) {
    console.warn(err);
  }
  broadcastPvp({
    type: "final",
    score: state.score,
    target: state.currentTarget?.label || "",
    message: `Final score ${state.score}`,
  });
  await refreshProfileAndLeaderboard();
  renderProfile();
  renderLeaderboard();
  refreshRetrainStatus();
  setStatus(
    `${message} Score: ${state.score}. Accuracy: ${accuracy.toFixed(1)}%. Skill Profile và Leaderboard đã cập nhật.`,
    "ok",
  );
}

function stopAllLoops() {
  clearInterval(state.timerId);
  clearInterval(state.realtimeId);
  state.timerId = null;
  state.realtimeId = null;
}


function updateRecognizeButton() {
  const recognizeBtn = $("#recognizeBtn");
  const generateBtn = $("#generateImageBtn");
  const isDrawingNow = Boolean(state.currentStroke);
  const canUseDrawing = Boolean(state.running && state.currentTarget && state.hasDrawn && !isDrawingNow);
  if (recognizeBtn) {
    recognizeBtn.disabled = !canUseDrawing || state.predictInFlight;
    recognizeBtn.textContent = state.predictInFlight ? "Đang nhận diện..." : "Nhận diện";
  }
  if (generateBtn) {
    generateBtn.disabled = !canUseDrawing || state.referenceImageInFlight;
    generateBtn.textContent = state.referenceImageInFlight ? "Đang sinh..." : "Sinh hình thật";
    generateBtn.title = isDrawingNow
      ? "Hãy nhấc bút/dừng vẽ rồi mới sinh hình thật"
      : "Sinh ảnh thật theo nhãn AI vừa nhận diện, hoặc theo từ mục tiêu nếu chưa nhận diện";
  }
}

async function manualGenerateReferenceImage() {
  if (!state.running || !state.currentTarget) {
    setStatus("Bấm Bắt đầu trước khi sinh hình thật.", "error");
    return;
  }
  if (!state.hasDrawn) {
    setStatus("Bạn hãy vẽ xong rồi mới bấm Sinh hình thật.", "error");
    return;
  }
  if (state.currentStroke) {
    setStatus("Hãy nhấc bút/dừng vẽ rồi mới bấm Sinh hình thật.", "error");
    return;
  }
  if (state.referenceImageInFlight) return;
  const label = state.prediction?.label || state.currentTarget?.label;
  if (!label) {
    setStatus("Chưa có nhãn để sinh hình thật.", "error");
    return;
  }
  const reason = state.prediction?.label ? "ai-label-manual" : "target-manual";
  const sourceText = state.prediction?.label ? "nhãn AI vừa nhận diện" : "từ cần vẽ";
  setStatus(`Đang sinh hình thật cho '${label}' theo ${sourceText}...`, "ok");
  await generateReferenceImage(label, reason);
}

function resetReferenceImage(message = "Vẽ xong rồi bấm Sinh hình thật để tạo ảnh. Hệ thống sẽ không tự sinh ảnh trong lúc bạn vẽ.") {
  state.referenceImageRequestId += 1;
  state.referenceImage = null;
  state.referenceImageInFlight = false;
  state.referenceImageMessage = message;
  updateReferenceImagePanel();
}

function updateReferenceImagePanel() {
  const box = $("#realImageBox");
  const badge = $("#realImageBadge");
  if (!box) return;
  if (state.referenceImageInFlight) {
    if (badge) {
      badge.textContent = "Đang sinh";
      badge.className = "badge warn";
    }
    box.className = "real-image-box loading";
    box.innerHTML = "Đang sinh ảnh thật theo yêu cầu của bạn...";
    return;
  }
  const data = state.referenceImage;
  if (!data?.image) {
    if (badge) {
      badge.textContent = "Chờ bấm";
      badge.className = "badge warn";
    }
    box.className = "real-image-box empty";
    box.textContent = state.referenceImageMessage || "Vẽ xong rồi bấm Sinh hình thật để tạo ảnh. Hệ thống sẽ không tự sinh ảnh trong lúc bạn vẽ.";
    return;
  }
  if (badge) {
    badge.textContent = data.label || "Ảnh thật";
    badge.className = "badge";
  }
  box.className = "real-image-box";
  box.innerHTML = `
    <img src="${data.image}" alt="Ảnh thật của ${escapeHtml(data.label || "từ vựng")}" />
    <div class="real-image-meta">
      <strong>${escapeHtml(data.label || "")}</strong>
      <span>${escapeHtml(data.meaning_vi || "")}</span>
    </div>
  `;
}

async function generateReferenceImage(label, reason = "target-after-draw") {
  if (!label) return;
  const requestId = state.referenceImageRequestId + 1;
  state.referenceImageRequestId = requestId;
  state.referenceImageInFlight = true;
  state.referenceImage = null;
  state.referenceImageMessage = "Đang sinh ảnh thật...";
  updateReferenceImagePanel();
  updateRecognizeButton();
  try {
    const form = new FormData();
    form.append("label", label);
    form.append("reason", reason);
    form.append("target", state.currentTarget?.label || "");
    form.append("predicted", state.prediction?.label || "");
    const data = await apiPostForm("/image/generate", form);
    if (requestId !== state.referenceImageRequestId) return;
    state.referenceImage = { ...data, reason };
    state.referenceImageInFlight = false;
    updateReferenceImagePanel();
    updateRecognizeButton();
  } catch (err) {
    if (requestId !== state.referenceImageRequestId) return;
    state.referenceImage = null;
    state.referenceImageInFlight = false;
    state.referenceImageMessage = `Không sinh được ảnh thật: ${err.message}`;
    updateReferenceImagePanel();
    updateRecognizeButton();
  }
}

function updateHud() {
  const target = state.currentTarget?.label || "---";
  $("#hudTarget").textContent = target;
  $("#targetChip").textContent = target;
  $("#hudScore").textContent = state.score;
  $("#hudTime").textContent = state.timeLeft;
  $("#hudLevel").textContent = `${state.level}/${state.gamePool.length || 40}`;
  $("#hudLives").textContent = "♥".repeat(Math.max(0, state.lives)) || "0";
  $("#hudStreak").textContent = state.streak;
}

function getIllustrationSVG(label) {
  const S = (paths, extra = "") =>
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" fill="none" stroke="#55e6a5" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" ${extra}>${paths}</svg>`;
  const map = {
    apple: S(
      `<circle cx="50" cy="58" r="26"/><path d="M50 32 Q55 20 65 18"/><path d="M50 34 Q44 24 36 26"/>`,
    ),
    baseball: S(
      `<circle cx="50" cy="50" r="28"/><path d="M34 26 Q38 38 34 50 Q30 62 34 74" stroke-width="3"/><path d="M66 26 Q62 38 66 50 Q70 62 66 74" stroke-width="3"/>`,
    ),
    book: S(
      `<rect x="22" y="18" width="56" height="64" rx="3"/><line x1="50" y1="18" x2="50" y2="82"/><line x1="30" y1="32" x2="48" y2="32"/><line x1="30" y1="42" x2="48" y2="42"/><line x1="30" y1="52" x2="48" y2="52"/>`,
    ),
    bowtie: S(
      `<polygon points="20,28 50,50 20,72" fill="rgba(85,230,165,0.15)"/><polygon points="80,28 50,50 80,72" fill="rgba(85,230,165,0.15)"/><circle cx="50" cy="50" r="6" fill="#55e6a5"/>`,
    ),
    diamond: S(
      `<polygon points="50,14 84,50 50,86 16,50" fill="rgba(85,230,165,0.12)"/><line x1="16" y1="50" x2="50" y2="14"/><line x1="50" y1="14" x2="84" y2="50"/><line x1="84" y1="50" x2="50" y2="86"/><line x1="50" y1="86" x2="16" y2="50"/><line x1="16" y1="50" x2="84" y2="50"/>`,
    ),
    dog: S(
      `<circle cx="50" cy="54" r="24"/><ellipse cx="30" cy="40" rx="10" ry="14" transform="rotate(-15,30,40)"/><ellipse cx="70" cy="40" rx="10" ry="14" transform="rotate(15,70,40)"/><circle cx="43" cy="50" r="3" fill="#55e6a5"/><circle cx="57" cy="50" r="3" fill="#55e6a5"/><ellipse cx="50" cy="60" rx="7" ry="4"/>`,
    ),
    door: S(
      `<rect x="28" y="14" width="44" height="72" rx="2"/><circle cx="64" cy="52" r="3" fill="#55e6a5"/><path d="M28 86 H72"/>`,
    ),
    envelope: S(
      `<rect x="14" y="28" width="72" height="48" rx="3"/><path d="M14 28 L50 58 L86 28"/>`,
    ),
    eye: S(
      `<ellipse cx="50" cy="50" rx="36" ry="22"/><circle cx="50" cy="50" r="12"/><circle cx="50" cy="50" r="6" fill="rgba(85,230,165,0.3)"/>`,
    ),
    fish: S(
      `<ellipse cx="46" cy="50" rx="24" ry="16"/><polygon points="70,50 86,36 86,64" fill="rgba(85,230,165,0.15)"/><circle cx="36" cy="46" r="3" fill="#55e6a5"/><line x1="30" y1="50" x2="58" y2="40" stroke-width="2"/>`,
    ),
    hat: S(
      `<path d="M20 70 Q50 60 80 70"/><path d="M35 70 Q36 38 50 34 Q64 38 65 70"/><rect x="15" y="68" width="70" height="8" rx="4"/>`,
    ),
    leaf: S(
      `<path d="M50 82 Q22 60 24 34 Q36 18 50 18 Q64 18 76 34 Q78 60 50 82Z" fill="rgba(85,230,165,0.15)"/><line x1="50" y1="82" x2="50" y2="22"/><line x1="50" y1="42" x2="36" y2="56" stroke-width="2"/><line x1="50" y1="54" x2="38" y2="65" stroke-width="2"/><line x1="50" y1="42" x2="64" y2="56" stroke-width="2"/><line x1="50" y1="54" x2="62" y2="65" stroke-width="2"/>`,
    ),
    lightning: S(
      `<polygon points="58,12 36,52 52,52 42,88 72,44 54,44" fill="rgba(255,209,102,0.2)" stroke="#ffd166" stroke-width="3"/>`,
    ),
    moon: S(
      `<path d="M70 28 Q86 50 70 72 Q50 82 34 72 Q54 68 58 50 Q54 32 34 28 Q50 18 70 28Z" fill="rgba(85,230,165,0.15)"/>`,
    ),
    pants: S(
      `<rect x="28" y="18" width="44" height="20" rx="3"/><path d="M28 38 L28 82 L50 82 L50 56 L50 82 L72 82 L72 38"/>`,
    ),
    scissors: S(
      `<line x1="50" y1="50" x2="24" y2="22"/><line x1="50" y1="50" x2="76" y2="22"/><line x1="50" y1="50" x2="28" y2="80"/><line x1="50" y1="50" x2="72" y2="80"/><circle cx="30" cy="78" r="10"/><circle cx="70" cy="78" r="10"/>`,
    ),
    square: S(`<rect x="18" y="18" width="64" height="64" rx="2"/>`),
    star: S(
      `<polygon points="50,12 61,37 88,37 67,56 75,82 50,64 25,82 33,56 12,37 39,37" fill="rgba(85,230,165,0.15)"/>`,
    ),
    "t-shirt": S(
      `<path d="M20 22 L36 14 L50 24 L64 14 L80 22 L70 44 L60 40 L60 82 L40 82 L40 40 L30 44Z" fill="rgba(85,230,165,0.12)"/>`,
    ),
    cat: S(
      `<circle cx="50" cy="56" r="22"/><polygon points="32,36 26,14 44,28" fill="rgba(85,230,165,0.2)"/><polygon points="68,36 74,14 56,28" fill="rgba(85,230,165,0.2)"/><circle cx="43" cy="52" r="3" fill="#55e6a5"/><circle cx="57" cy="52" r="3" fill="#55e6a5"/><path d="M42 62 Q50 67 58 62"/><line x1="30" y1="58" x2="50" y2="62" stroke-width="2"/><line x1="70" y1="58" x2="50" y2="62" stroke-width="2"/>`,
    ),
    sun: S(
      `<circle cx="50" cy="50" r="18" fill="rgba(255,209,102,0.2)" stroke="#ffd166"/><line x1="50" y1="10" x2="50" y2="22" stroke="#ffd166"/><line x1="50" y1="78" x2="50" y2="90" stroke="#ffd166"/><line x1="10" y1="50" x2="22" y2="50" stroke="#ffd166"/><line x1="78" y1="50" x2="90" y2="50" stroke="#ffd166"/><line x1="22" y1="22" x2="30" y2="30" stroke="#ffd166"/><line x1="70" y1="70" x2="78" y2="78" stroke="#ffd166"/><line x1="78" y1="22" x2="70" y2="30" stroke="#ffd166"/><line x1="22" y1="78" x2="30" y2="70" stroke="#ffd166"/>`,
    ),
    tree: S(
      `<ellipse cx="50" cy="36" rx="28" ry="24" fill="rgba(85,230,165,0.15)"/><rect x="44" y="58" width="12" height="26" rx="2"/>`,
    ),
    flower: S(
      `<circle cx="50" cy="50" r="10" fill="rgba(255,209,102,0.3)" stroke="#ffd166"/><ellipse cx="50" cy="28" rx="9" ry="14" fill="rgba(85,230,165,0.2)"/><ellipse cx="50" cy="72" rx="9" ry="14" fill="rgba(85,230,165,0.2)"/><ellipse cx="28" cy="50" rx="14" ry="9" fill="rgba(85,230,165,0.2)"/><ellipse cx="72" cy="50" rx="14" ry="9" fill="rgba(85,230,165,0.2)"/><ellipse cx="34" cy="34" rx="9" ry="14" fill="rgba(85,230,165,0.2)" transform="rotate(-45,34,34)"/><ellipse cx="66" cy="34" rx="9" ry="14" fill="rgba(85,230,165,0.2)" transform="rotate(45,66,34)"/><ellipse cx="34" cy="66" rx="9" ry="14" fill="rgba(85,230,165,0.2)" transform="rotate(45,34,66)"/><ellipse cx="66" cy="66" rx="9" ry="14" fill="rgba(85,230,165,0.2)" transform="rotate(-45,66,66)"/>`,
    ),
    cloud: S(
      `<ellipse cx="40" cy="58" rx="24" ry="18"/><ellipse cx="62" cy="58" rx="20" ry="16"/><ellipse cx="50" cy="46" rx="18" ry="16"/>`,
    ),
    umbrella: S(
      `<path d="M14 50 Q14 20 50 18 Q86 20 86 50Z"/><line x1="50" y1="18" x2="50" y2="76"/><path d="M50 76 Q50 86 40 86 Q30 86 30 76"/>`,
    ),
    key: S(
      `<circle cx="36" cy="42" r="16"/><circle cx="36" cy="42" r="8"/><line x1="50" y1="42" x2="86" y2="42"/><line x1="74" y1="42" x2="74" y2="54"/><line x1="84" y1="42" x2="84" y2="52"/>`,
    ),
    cup: S(
      `<path d="M26 30 L34 80 L66 80 L74 30Z"/><path d="M74 46 Q88 46 88 58 Q88 70 74 70"/>`,
    ),
    clock: S(
      `<circle cx="50" cy="50" r="34"/><line x1="50" y1="50" x2="50" y2="24" stroke-width="5"/><line x1="50" y1="50" x2="68" y2="60" stroke-width="3"/><circle cx="50" cy="50" r="3" fill="#55e6a5"/>`,
    ),
    car: S(
      `<rect x="12" y="44" width="76" height="30" rx="6"/><path d="M24 44 L32 24 L68 24 L76 44"/><circle cx="28" cy="76" r="10"/><circle cx="72" cy="76" r="10"/><rect x="36" y="28" width="14" height="14" rx="2"/><rect x="52" y="28" width="14" height="14" rx="2"/>`,
    ),
    bicycle: S(
      `<circle cx="28" cy="62" r="20"/><circle cx="72" cy="62" r="20"/><line x1="28" y1="62" x2="50" y2="32"/><line x1="72" y1="62" x2="50" y2="32"/><line x1="50" y1="32" x2="28" y2="62"/><line x1="50" y1="32" x2="50" y2="48"/><line x1="44" y1="26" x2="56" y2="26"/>`,
    ),
    airplane: S(
      `<path d="M8 54 L50 36 L92 54 L74 54 L74 68 L50 62 L26 68 L26 54Z" fill="rgba(85,230,165,0.12)"/><path d="M50 36 L50 76"/><path d="M42 72 L58 72"/>`,
    ),
    house: S(
      `<polygon points="50,14 88,46 78,46 78,86 22,86 22,46 12,46" fill="rgba(85,230,165,0.12)"/><rect x="38" y="62" width="24" height="24" rx="2"/>`,
    ),
    banana: S(
      `<path d="M26 78 Q18 50 32 28 Q46 16 66 18 Q74 18 74 26 Q74 34 60 34 Q40 34 32 52 Q26 66 34 78Z" fill="rgba(255,209,102,0.2)" stroke="#ffd166"/>`,
    ),
    "ice cream": S(
      `<polygon points="50,88 26,42 74,42" fill="rgba(85,230,165,0.12)"/><circle cx="50" cy="34" r="18" fill="rgba(255,120,80,0.2)" stroke="#ff7850" stroke-width="3"/><path d="M38 28 Q44 22 50 28 Q56 22 62 28" stroke="#ff7850" stroke-width="2"/>`,
    ),
    cake: S(
      `<rect x="20" y="52" width="60" height="34" rx="4" fill="rgba(85,230,165,0.12)"/><rect x="30" y="42" width="40" height="12" rx="2"/><line x1="34" y1="22" x2="34" y2="42"/><line x1="50" y1="18" x2="50" y2="42"/><line x1="66" y1="22" x2="66" y2="42"/><ellipse cx="34" cy="20" rx="4" ry="6" stroke="#ffd166"/><ellipse cx="50" cy="16" rx="4" ry="6" stroke="#ffd166"/><ellipse cx="66" cy="20" rx="4" ry="6" stroke="#ffd166"/>`,
    ),
    candle: S(
      `<rect x="38" y="42" width="24" height="46" rx="3" fill="rgba(85,230,165,0.12)"/><path d="M50 12 Q54 22 50 30 Q46 22 50 12Z" fill="rgba(255,209,102,0.4)" stroke="#ffd166"/><line x1="50" y1="30" x2="50" y2="42"/>`,
    ),
    guitar: S(
      `<path d="M50 16 Q58 16 60 24 L62 52 Q68 54 68 62 Q68 76 50 82 Q32 76 32 62 Q32 54 38 52 L40 24 Q42 16 50 16Z" fill="rgba(85,230,165,0.12)"/><circle cx="50" cy="62" r="8"/><line x1="42" y1="16" x2="58" y2="16"/>`,
    ),
    hammer: S(
      `<line x1="42" y1="50" x2="70" y2="82"/><rect x="34" y="22" width="36" height="22" rx="4" fill="rgba(85,230,165,0.15)" transform="rotate(-30,52,33)"/>`,
    ),
    bed: S(
      `<rect x="10" y="48" width="80" height="34" rx="4"/><rect x="10" y="44" width="18" height="38" rx="3"/><rect x="72" y="44" width="18" height="38" rx="3"/><ellipse cx="38" cy="48" rx="18" ry="10" fill="rgba(85,230,165,0.2)"/><ellipse cx="62" cy="48" rx="18" ry="10" fill="rgba(85,230,165,0.2)"/>`,
    ),
    chair: S(
      `<rect x="28" y="34" width="44" height="8" rx="2"/><line x1="28" y1="42" x2="24" y2="82"/><line x1="72" y1="42" x2="76" y2="82"/><line x1="32" y1="42" x2="32" y2="82"/><line x1="68" y1="42" x2="68" y2="82"/><rect x="28" y="14" width="8" height="22" rx="2"/><rect x="64" y="14" width="8" height="22" rx="2"/><line x1="28" y1="24" x2="72" y2="24"/>`,
    ),
  };
  return (
    map[label] ||
    S(
      `<text x="50" y="56" text-anchor="middle" font-size="36" fill="#55e6a5" stroke="none">${label.slice(0, 2).toUpperCase()}</text>`,
    )
  );
}

function updateTargetPanel() {
  const item = state.currentTarget;
  if (!item) return;
  $("#targetWord").textContent = item.label;
  $("#wordMeta").innerHTML = `
    <p><b>Nghĩa:</b> ${escapeHtml(item.meaning_vi || item.label)}</p>
    <p><b>IPA:</b> ${escapeHtml(item.ipa || "")}</p>
    <p><b>Ví dụ:</b> ${escapeHtml(item.example_en || "")}</p>
    <p><b>Dịch:</b> ${escapeHtml(item.example_vi || "")}</p>
  `;
}

function updatePredictionPanel(data) {
  const box = $("#predictionList");
  if (!data || !data.top5) {
    box.textContent = "AI chưa có dữ liệu. Vẽ xong bấm Nhận diện.";
    return;
  }
  const conf = Number(data.confidence || 0);
  const target = state.currentTarget?.label || data.target || "";
  const threshold = Math.max(
    0.58,
    SUCCESS_THRESHOLD_BASE - Math.min(state.streak, 8) * 0.01,
  );
  const match = data.label === target;
  const source = data.ai_source || data.source || "image-cnn";
  const faceRow = state.mode === "camera" && state.cameraTool === "face"
    ? `<div class="mini-row"><span>Face input</span><small>${escapeHtml(cameraFaceLabel())}</small></div>`
    : "";
  const rerank = data.rerank || data.judge?.rerank || null;
  const rerankRow = rerank?.used
    ? `<div class="mini-row"><span>Tối ưu nhận diện</span><small>rerank hình học · CNN gốc: ${escapeHtml(rerank.raw_label || "?")} ${Math.round(Number(rerank.raw_confidence || 0) * 100)}%</small></div>`
    : rerank?.target_shape_score
      ? `<div class="mini-row"><span>Shape check</span><small>${Math.round(Number(rerank.target_shape_score || 0) * 100)}% · chưa đủ để sửa CNN</small></div>`
      : "";
  const sourceRows = `
    <div class="mini-row"><span>AI source</span><small>${escapeHtml(source)}</small></div>
    ${faceRow}
    ${rerankRow}
    <div class="mini-row"><span>${match ? "Đúng mục tiêu" : "Đang lệch mục tiêu"}</span><small>${Math.round(conf * 100)}% / cần ${Math.round(threshold * 100)}%</small></div>
    <div class="mini-row"><span>Chế độ</span><small>Nhận diện thủ công bằng nút</small></div>
  `;
  const rows = data.top5
    .map((p) => {
      const pct = Math.round(Number(p.confidence || 0) * 100);
      const isTarget = p.label === target;
      return `
        <div class="pred-row ${isTarget ? "target-hit" : ""}">
          <div class="pred-label">${isTarget ? "✓ " : ""}${escapeHtml(p.label)}</div>
          <div class="pred-bar"><span style="width:${pct}%"></span></div>
          <div>${pct}%</div>
        </div>
      `;
    })
    .join("");
  box.innerHTML = sourceRows + rows;
}

function updateJudge(judge) {
  const badge = $("#gradeBadge");
  badge.textContent = judge?.grade || "---";
  badge.className = `badge ${judge?.correct ? "" : "warn"}`.trim();
  $("#shapeScore").textContent = judge?.shape_score ?? 0;
  $("#clarityScore").textContent = judge?.clarity_score ?? 0;
  $("#strokeScore").textContent = judge?.stroke_score ?? 0;
  $("#speedScore").textContent = judge?.speed_score ?? 0;
  if (!judge) {
    $("#judgeFeedback").textContent = "AI sẽ đánh giá khi bạn bấm Nhận diện.";
    return;
  }
  const verdict = judge.correct
    ? `AI đã nhận đúng '${judge.target}'.`
    : `AI đoán '${judge.predicted}', mục tiêu là '${judge.target}'.`;
  $("#judgeFeedback").textContent = `${verdict} ${judge.feedback || ""}`;
}

function renderProfile() {
  const profile = state.profile;
  const stats = profile?.stats || {};
  const training = profile?.training || {};
  $("#profileBox").innerHTML = `
    <div class="profile-item"><span>Games</span><strong>${stats.games || 0}</strong></div>
    <div class="profile-item"><span>Best score</span><strong>${stats.best_score || 0}</strong></div>
    <div class="profile-item"><span>Drawings</span><strong>${stats.drawings || 0}</strong></div>
    <div class="profile-item"><span>Accuracy</span><strong>${stats.accuracy || 0}%</strong></div>
    <div class="profile-item"><span>Best streak</span><strong>${stats.best_streak || 0}</strong></div>
    <div class="profile-item"><span>Avg conf</span><strong>${stats.avg_confidence || 0}%</strong></div>
  `;
  const weaknesses = profile?.weaknesses || [];
  const plan = profile?.practice_plan || [];
  const readiness = `<div class="mini-row"><span>Train readiness</span><small>${training.ready_stroke ? "stroke ready" : "cần thêm mẫu"} · ${training.total_samples || 0} mẫu/${training.classes || 0} lớp</small></div>`;
  const weakRows = weaknesses.length
    ? weaknesses
        .map(
          (w) =>
            `<div class="mini-row column"><span>${escapeHtml(w.target)} <em>${escapeHtml(w.band || "")}</em></span><small>${w.accuracy}% / ${w.attempts} lần · ${escapeHtml(w.tip || "")}</small></div>`,
        )
        .join("")
    : `<div class="mini-row"><span>Chưa có dữ liệu yếu/mạnh</span><small>Chơi vài vòng để cập nhật</small></div>`;
  const planRows = plan.length
    ? `<div class="mini-title">Practice plan</div>` +
      plan
        .map(
          (p) =>
            `<div class="mini-row column"><span>${escapeHtml(p.target)} · mục tiêu ${escapeHtml(String(p.accuracy))}%</span><small>${escapeHtml(p.goal)} ${escapeHtml(p.tip || "")}</small></div>`,
        )
        .join("")
    : "";
  $("#weaknessBox").innerHTML = readiness + weakRows + planRows;
}

function renderLeaderboard() {
  const rows = state.leaderboard || [];
  $("#leaderboardBox").innerHTML = rows.length
    ? rows
        .map((r, i) => {
          const mine = r.username === state.user?.username;
          return `<div class="mini-row ${mine ? "mine" : ""}"><span>#${i + 1} ${escapeHtml(r.username || "guest")}</span><small>${r.score || 0} pts · L${r.level || 0} · streak ${r.streak || 0}</small></div>`;
        })
        .join("")
    : `<div class="mini-row"><span>Chưa có bảng xếp hạng</span><small>Hãy chơi 1 lượt</small></div>`;
}

function setStatus(text, type = "") {
  const el = $("#gameStatus");
  if (!el) return;
  el.textContent = text;
  el.className = `status ${type}`.trim();
}

function showCorrectFlash(text) {
  const flash = $("#correctFlash");
  flash.textContent = text;
  flash.classList.add("show");
  setTimeout(() => flash.classList.remove("show"), 720);
}

function getAllCurrentStrokes() {
  return state.strokes.concat(state.currentStroke ? [state.currentStroke] : []);
}

function countStrokePoints() {
  return getAllCurrentStrokes().reduce((sum, stroke) => sum + stroke.length, 0);
}

async function saveStrokeSample(correct = false, manual = false) {
  if (!state.currentTarget) return null;
  const strokes = getAllCurrentStrokes();
  const pointCount = countStrokePoints();
  if (!pointCount) {
    if (manual) setStatus("Chưa có nét vẽ để lưu mẫu train.", "error");
    return null;
  }
  const key = `${state.currentTarget.label}|${state.level}|${pointCount}|${correct ? 1 : 0}|${manual ? 1 : 0}`;
  if (!manual && state.lastSavedKey === key) return null;
  state.lastSavedKey = key;

  const form = new FormData();
  form.append("target", state.currentTarget.label);
  form.append("predicted", state.prediction?.label || "");
  form.append("confidence", String(state.prediction?.confidence || 0));
  form.append("correct", correct ? "1" : "0");
  form.append("mode", currentInputMode());
  form.append("strokes_json", JSON.stringify(strokes));
  form.append("judge_json", JSON.stringify(state.judge || {}));
  form.append("manual", manual ? "1" : "0");
  form.append("point_count", String(pointCount));
  try {
    const data = await apiPostForm("/game/stroke", form);
    await refreshProfileAndLeaderboard();
    renderProfile();
    setStatus(
      correct
        ? `Đã lưu mẫu đúng #${data.sample_id} cho '${data.target}' (${data.point_count} điểm).`
        : `Đã lưu mẫu train #${data.sample_id} cho '${data.target}' để AI học lỗi/khó.`,
      correct ? "ok" : "",
    );
    return data;
  } catch (err) {
    console.warn(err);
    if (manual) setStatus(`Lưu mẫu lỗi: ${err.message}`, "error");
    return null;
  }
}

async function tryStrokePrediction() {
  if (!state.currentTarget || countStrokePoints() < 8) return null;
  try {
    const form = new FormData();
    form.append("strokes_json", JSON.stringify(getAllCurrentStrokes()));
    form.append("target", state.currentTarget.label);
    const data = await apiPostForm("/predict_stroke", form);
    state.strokeModelAvailable = Boolean(data.available);
    return data;
  } catch (err) {
    return null;
  }
}

function togglePvp() {
  if (state.pvpConnected) {
    disconnectPvp();
    return;
  }
  connectPvp();
}

function connectPvp() {
  const input = $("#pvpRoomInput");
  state.pvpRoom = (input?.value || "airdraw").trim().toLowerCase() || "airdraw";
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const url = `${protocol}://${location.host}/ws/pvp/${encodeURIComponent(state.pvpRoom)}?username=${encodeURIComponent(state.user?.username || "guest")}`;
  try {
    state.pvpSocket = new WebSocket(url);
    state.pvpSocket.onopen = () => {
      state.pvpConnected = true;
      renderPvp();
      broadcastPvp({
        type: "hello",
        score: state.score,
        target: state.currentTarget?.label || "",
      });
    };
    state.pvpSocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        state.pvpEvents.unshift(data);
        state.pvpEvents = state.pvpEvents.slice(0, 5);
        renderPvp(data.players || []);
      } catch {}
    };
    state.pvpSocket.onclose = () => {
      state.pvpConnected = false;
      renderPvp();
    };
  } catch (err) {
    setStatus(`Không kết nối được PvP: ${err.message}`, "error");
  }
}

function disconnectPvp() {
  if (state.pvpSocket) state.pvpSocket.close();
  state.pvpSocket = null;
  state.pvpConnected = false;
  renderPvp();
}

function broadcastPvp(payload) {
  if (
    !state.pvpConnected ||
    !state.pvpSocket ||
    state.pvpSocket.readyState !== WebSocket.OPEN
  )
    return;
  state.pvpSocket.send(
    JSON.stringify({
      ...payload,
      score: state.score,
      level: state.level,
      ts: Date.now(),
    }),
  );
}

function renderPvp(players = []) {
  const badge = $("#pvpBadge");
  const btn = $("#pvpBtn");
  const box = $("#pvpBox");
  if (!badge || !btn || !box) return;
  badge.textContent = state.pvpConnected ? "ON" : "OFF";
  badge.className = state.pvpConnected ? "badge" : "badge warn";
  btn.textContent = state.pvpConnected ? "Leave" : "Join";
  const playerRows = players.length
    ? players
        .map(
          (p, i) =>
            `<div class="mini-row"><span>#${i + 1} ${escapeHtml(p.username || "guest")}</span><small>${p.score || 0} pts · L${p.level || 1} · ${escapeHtml(p.target || "waiting")}</small></div>`,
        )
        .join("")
    : `<div class="mini-row"><span>${state.pvpConnected ? "Đang chờ người chơi" : "Chưa vào phòng"}</span><small>${escapeHtml(state.pvpRoom)}</small></div>`;
  const eventRows = state.pvpEvents
    .slice(0, 3)
    .map(
      (e) =>
        `<div class="mini-row"><span>${escapeHtml(e.username || "system")}</span><small>${escapeHtml(e.message || e.label || e.type || "event")}</small></div>`,
    )
    .join("");
  box.innerHTML = playerRows + eventRows;
}

async function exportDataset() {
  // Chạy im lặng: chỉ export & lưu, không đổ log ra panel.
  setStatus("Đang export dữ liệu...", "");
  try {
    const data = await apiGet("/dataset/export");
    await refreshProfileAndLeaderboard();
    renderProfile();
    setStatus(`Đã export & lưu ${data.samples || 0} mẫu, ${data.classes || 0} lớp.`, "ok");
  } catch (err) {
    setStatus("Export lỗi: " + err.message, "warn");
  }
}

async function startRetrain(mode) {
  // Chạy im lặng: bắt đầu train & tự lưu, không đổ log ra panel.
  const form = new FormData();
  form.append("mode", mode);
  form.append("epochs", mode === "stroke" ? "8" : "5");
  state.imageModelReloaded = false;
  setStatus(`Đang train ${mode}...`, "");
  try {
    const data = await apiPostForm("/admin/retrain/start", form);
    setStatus(data.ok ? `Đã bắt đầu train ${mode}.` : "Hệ thống đang bận train.", data.ok ? "ok" : "warn");
    setTimeout(refreshRetrainStatus, 1500);
  } catch (err) {
    setStatus("Train lỗi: " + err.message, "warn");
  }
}

async function reloadImageModel() {
  const form = new FormData();
  form.append("kind", "self_improved");
  try {
    const data = await apiPostForm("/admin/model/reload", form);
    state.imageModelReloaded = true;
    await loadGameData();
    updateHud();
    renderProfile();
    setStatus(`Đã reload model ảnh mới (${data.model?.num_recognition_categories || "?"} lớp).`, "ok");
    return data;
  } catch (err) {
    console.warn(err);
    return null;
  }
}

async function refreshRetrainStatus() {
  // Theo dõi train ngầm để tự reload model khi xong, KHÔNG đổ log ra panel.
  try {
    const data = await apiGet("/admin/retrain/status");
    state.retrainStatus = data;
    if (data.status === "done") {
      if (data.mode === "image" && data.self_improved_model_exists && !state.imageModelReloaded) {
        await reloadImageModel();
      }
      setStatus(`Train ${data.mode || ""} xong & đã lưu.`, "ok");
    }
    if (data.process_running) setTimeout(refreshRetrainStatus, 2500);
  } catch (err) {
    // im lặng
  }
}



async function showPanelStorage(section = "") {
  // Theo yêu cầu: bấm nút chỉ tự động lưu dữ liệu, KHÔNG hiển thị log/danh sách file.
  const box = $("#panelStorageBox");
  if (box) box.innerHTML = "";
  try {
    const qs = section ? `?section=${encodeURIComponent(section)}&limit=1` : "?limit=1";
    await apiGet(`/admin/panel-storage${qs}`); // chạm endpoint để chắc chắn dữ liệu đã được ghi
  } catch (err) {
    // im lặng, không hiển thị log lỗi ra panel
  }
  setStatus(
    section === "real_image_after_draw" ? "Đã lưu dữ liệu hình thật." : "Đã lưu toàn bộ dữ liệu panel.",
    "ok",
  );
}

async function showSelfLoopStorage() {
  // Bấm là tự lưu, không hiển thị log.
  try {
    await apiGet("/admin/self-improve/storage");
  } catch (err) {
    // im lặng
  }
  setStatus("Đã lưu dữ liệu self-improving loop.", "ok");
}


async function refreshProStatus() {
  const box = $("#proOpsBox");
  if (!box) return;
  try {
    const data = await apiGet("/admin/pro/status");
    state.proStatus = data;
    const bm = data.benchmark || {};
    const ev = data.latest_eval || {};
    const ready = data.training_readiness || {};
    const cal = data.calibration || {};
    const promo = (data.promotion_tail || [])[data.promotion_tail?.length - 1] || null;
    const bmText = bm.num_samples
      ? `${bm.num_samples} mẫu · ${bm.num_classes} lớp · min/class ${bm.min_samples_per_class || 0}`
      : "chưa build";
    const evalText = ev.samples
      ? `top1 ${Math.round((ev.top1_accuracy || 0) * 100)}% · F1 ${Math.round((ev.macro_f1 || 0) * 100)}% · ECE ${Math.round((ev.ece_10_bins || 0) * 100)}%`
      : "chưa evaluate";
    const calText = cal.temperature
      ? `T=${Number(cal.temperature).toFixed(2)} · ECE ${Math.round((cal.ece_after || 0) * 100)}%`
      : "chưa calibration";
    box.innerHTML = `
      <div class="mini-row"><span>Benchmark</span><small>${escapeHtml(bmText)}</small></div>
      <div class="mini-row"><span>Eval release</span><small>${escapeHtml(evalText)}</small></div>
      <div class="mini-row"><span>Calibration</span><small>${escapeHtml(calText)}</small></div>
      <div class="mini-row"><span>Readiness</span><small>${ready.ready_stroke ? "stroke ready" : "cần thêm stroke"} · ${ready.ready_image ? "image ready" : "cần dữ liệu image"}</small></div>
      <div class="mini-row"><span>Promotion</span><small>${promo ? (promo.approved ? "approved" : "rejected/check") : "chưa chạy"}</small></div>
    `;
  } catch (err) {
    box.innerHTML = `<div class="mini-row column"><span>Production status lỗi</span><small>${escapeHtml(err.message)}</small></div>`;
  }
}

async function buildBenchmark() {
  const box = $("#proOpsBox");
  if (box) box.innerHTML = `<div class="mini-row"><span>Đang build benchmark</span><small>đọc stroke_samples...</small></div>`;
  try {
    const data = await apiPostForm("/admin/benchmark/build", new FormData());
    const m = data.manifest || {};
    if (box) box.innerHTML = `<div class="mini-row"><span>Benchmark OK</span><small>${m.num_samples || 0} mẫu · ${m.num_classes || 0} lớp</small></div>`;
    await refreshProStatus();
  } catch (err) {
    if (box) box.innerHTML = `<div class="mini-row column"><span>Build lỗi</span><small>${escapeHtml(err.message)}</small></div>`;
  }
}

async function evaluateRelease() {
  const box = $("#proOpsBox");
  if (box) box.innerHTML = `<div class="mini-row"><span>Đang evaluate</span><small>có thể mất vài phút nếu model lớn</small></div>`;
  try {
    const data = await apiPostForm("/admin/evaluate/run", new FormData());
    const s = data.summary || {};
    if (box) box.innerHTML = `<div class="mini-row"><span>Evaluate OK</span><small>top1 ${Math.round((s.top1_accuracy || 0) * 100)}% · ${s.samples || 0} mẫu</small></div>`;
    await refreshProStatus();
  } catch (err) {
    if (box) box.innerHTML = `<div class="mini-row column"><span>Evaluate lỗi</span><small>${escapeHtml(err.message)}</small></div>`;
  }
}

async function promoteDryRun() {
  const box = $("#proOpsBox");
  if (box) box.innerHTML = `<div class="mini-row"><span>Đang check promote</span><small>không thay model thật</small></div>`;
  try {
    const data = await apiPostForm("/admin/promote/dry-run", new FormData());
    if (box) box.innerHTML = `<div class="mini-row column"><span>Promote dry-run</span><small>${data.ok ? "Có thể promote theo gate prototype" : "Chưa đạt gate"}</small></div>`;
    await refreshProStatus();
  } catch (err) {
    if (box) box.innerHTML = `<div class="mini-row column"><span>Promote lỗi</span><small>${escapeHtml(err.message)}</small></div>`;
  }
}

function speakCurrentWord() {
  const item = state.currentTarget;
  if (!item || !window.speechSynthesis) return;
  const text = `${item.label}. ${item.example_en || ""}`;
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = "en-US";
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utter);
}


function loadScriptOnce(src, globalTest, timeoutMs = 9000) {
  if (globalTest()) return Promise.resolve(true);
  const existing = [...document.scripts].find((script) => script.src === src || script.src.includes(src.split("/").slice(-2).join("/")));
  if (existing) {
    return new Promise((resolve) => {
      existing.addEventListener("load", () => resolve(Boolean(globalTest())), { once: true });
      existing.addEventListener("error", () => resolve(false), { once: true });
      setTimeout(() => resolve(Boolean(globalTest())), timeoutMs);
    });
  }
  return new Promise((resolve) => {
    const script = document.createElement("script");
    script.src = src;
    script.defer = true;
    script.onload = () => resolve(Boolean(globalTest()));
    script.onerror = () => resolve(false);
    document.head.appendChild(script);
    setTimeout(() => resolve(Boolean(globalTest())), timeoutMs);
  });
}

async function ensureHandsReady() {
  const loaded = await loadScriptOnce(
    "https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js",
    () => Boolean(window.Hands),
  );
  if (!loaded) throw new Error("MediaPipe Hands could not be loaded");
  if (state.hands) return true;
  state.hands = new window.Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });
  state.hands.setOptions({
    maxNumHands: 1,
    // modelComplexity 0 = nhẹ & nhanh hơn nhiều trên trình duyệt -> nhiều khung
    // hình/giây hơn -> nét vẽ liền mạch, ít đứt khúc.
    modelComplexity: 0,
    // Hạ ngưỡng tracking để MediaPipe duy trì bám ngón tay liên tục giữa các
    // khung (ít mất tay giữa chừng -> ít gãy nét).
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.4,
  });
  state.hands.onResults(onHandResults);
  return true;
}

async function ensureFaceMeshReady() {
  const loaded = await loadScriptOnce(
    "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js",
    () => Boolean(window.FaceMesh),
  );
  if (!loaded) throw new Error("MediaPipe FaceMesh could not be loaded");
  if (state.faceMesh) return true;
  state.faceMesh = new window.FaceMesh({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
  });
  state.faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.65,
    minTrackingConfidence: 0.6,
  });
  state.faceMesh.onResults(onFaceResults);
  return true;
}

async function startCamera() {
  const video = $("#cameraVideo");
  const stage = $("#gameStage");
  if (!video) {
    setStatus("Không tìm thấy khung video camera. Hãy tải lại trang.", "error");
    return false;
  }

  // Camera (getUserMedia) CHỈ hoạt động ở "ngữ cảnh bảo mật": https://, hoặc
  // http://localhost / 127.0.0.1. Nếu mở bằng IP mạng LAN (vd http://192.168.x.x)
  // trình duyệt sẽ chặn và navigator.mediaDevices thường là undefined.
  const localHosts = ["localhost", "127.0.0.1", "[::1]", "::1"];
  const isLocal = localHosts.includes(location.hostname);
  if (!window.isSecureContext && !isLocal) {
    setStatus(
      `Trình duyệt CHẶN camera vì trang không chạy ở ngữ cảnh bảo mật ` +
        `(đang mở: ${location.origin}). Hãy mở game bằng http://localhost:8000 ` +
        `(KHÔNG dùng địa chỉ IP mạng LAN), hoặc bật HTTPS.`,
      "error",
    );
    return false;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus(
      "Trình duyệt không hỗ trợ camera API (getUserMedia). Hãy dùng Chrome/Edge " +
        "mới nhất và mở qua http://localhost:8000.",
      "error",
    );
    return false;
  }

  // Đảm bảo class camera-mode để CSS hiển thị video (phòng trường hợp toggle bị lệch)
  if (stage) stage.classList.add("camera-mode");
  stage?.classList.remove("mouse-mode");

  if (!state.cameraStream) {
    let stream = null;
    try {
      // Dùng ideal thay vì exact để tương thích tốt hơn với nhiều webcam
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 960 },
          height: { ideal: 540 },
          facingMode: "user",
        },
        audio: false,
      });
      video.srcObject = stream;
      // Đợi play xong, một số trình duyệt cần user gesture rõ ràng
      await video.play().catch((playErr) => {
        console.warn("video.play() warning:", playErr);
        // Vẫn tiếp tục, nhiều trường hợp stream vẫn chạy
      });
      state.cameraStream = stream;
      setStatus(cameraInstruction(), "ok");
    } catch (err) {
      // Dọn stream nếu đã xin được quyền nhưng play/src lỗi
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      state.cameraStream = null;

      console.error("getUserMedia error:", err.name, err.message, err);
      let msg = `Không mở được camera: ${err.name} - ${err.message || ""}`;
      if (
        err.name === "NotAllowedError" ||
        err.name === "PermissionDeniedError"
      ) {
        msg =
          "Bạn đã từ chối quyền camera HOẶC Windows đang chặn. Sửa: (1) bấm ổ khóa cạnh thanh địa chỉ -> Camera -> Allow; (2) Windows Settings -> Privacy & security -> Camera -> bật 'Camera access' và cho phép trình duyệt. Rồi tải lại trang.";
      } else if (
        err.name === "NotReadableError" ||
        err.name === "TrackStartError" ||
        err.name === "AbortError"
      ) {
        msg =
          "Camera đang bị ứng dụng khác chiếm dụng (Zoom/Teams/OBS/Camera app...) hoặc driver lỗi. Hãy ĐÓNG các app đó, rồi tải lại trang và thử lại.";
      } else if (
        err.name === "NotFoundError" ||
        err.name === "DevicesNotFoundError"
      ) {
        msg =
          "Không tìm thấy camera. Hãy cắm webcam hoặc dùng laptop có camera tích hợp.";
      } else if (err.name === "OverconstrainedError") {
        msg =
          "Camera không hỗ trợ độ phân giải yêu cầu. Đang thử lại với cấu hình đơn giản...";
        // Thử lại với cấu hình tối thiểu
        try {
          const fallbackStream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false,
          });
          video.srcObject = fallbackStream;
          await video.play().catch(() => {});
          state.cameraStream = fallbackStream;
          setStatus(
            "Camera bật (chế độ đơn giản). Tracking vẫn hoạt động.",
            "ok",
          );
          // Tiếp tục init MediaPipe bên dưới
        } catch (e2) {
          setStatus(msg, "error");
          return false;
        }
      } else {
        setStatus(msg, "error");
        return false;
      }
    }
  }

  try {
    if (state.cameraTool === "face") {
      state.handLoop = false;
      await ensureFaceMeshReady();
      if (state.faceMesh && !state.faceLoop) {
        state.faceLoop = true;
        faceFrameLoop();
      }
    } else {
      state.faceLoop = false;
      await ensureHandsReady();
      if (state.hands && !state.handLoop) {
        state.handLoop = true;
        handFrameLoop();
      }
    }
  } catch (err) {
    console.warn("MediaPipe init error:", err);
    setStatus(`${err.message}. Camera đã bật nhưng bộ nhận diện đang chọn chưa sẵn sàng.`, "error");
  }
  updateCameraToolUI();
  return true;
}

function stopCamera() {
  state.handLoop = false;
  state.faceLoop = false;
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((track) => track.stop());
    state.cameraStream = null;
  }
  const video = $("#cameraVideo");
  if (video) video.srcObject = null;
  $("#handCanvas")?.getContext("2d")?.clearRect(0, 0, CANVAS_W, CANVAS_H);
  $("#faceCanvas")?.getContext("2d")?.clearRect(0, 0, CANVAS_W, CANVAS_H);
  state.faceDrawingActive = false;
  state.faceFilteredPoint = null;
  if (typeof resetFaceEuro === "function") resetFaceEuro();
  state.faceCenter = null;
  state.faceStatus = null;
  state.filteredPoint = null;
  if (typeof resetHandEuro === "function") resetHandEuro();
  state.penLiftFrames = 0;
  state.lastMid = null;
  clearFaceOverlay();
}

async function waitNextFrame() {
  await new Promise((resolve) => requestAnimationFrame(resolve));
}

async function handFrameLoop() {
  while (state.handLoop && state.hands) {
    const video = $("#cameraVideo");
    if (!video || video.readyState < 2 || state.mode !== "camera" || state.cameraTool !== "hand") {
      await waitNextFrame();
      continue;
    }
    try {
      await state.hands.send({ image: video });
    } catch (err) {
      console.warn(err);
    }
    await waitNextFrame();
  }
}

async function faceFrameLoop() {
  while (state.faceLoop && state.faceMesh) {
    const video = $("#cameraVideo");
    if (!video || video.readyState < 2 || state.mode !== "camera" || state.cameraTool !== "face") {
      await waitNextFrame();
      continue;
    }
    try {
      await state.faceMesh.send({ image: video });
    } catch (err) {
      console.warn(err);
    }
    await waitNextFrame();
  }
}


function onFaceResults(results) {
  if (state.cameraTool !== "face") return;
  const canvas = $("#faceCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
  const landmarks = results.multiFaceLandmarks?.[0];
  if (!landmarks) {
    state.faceStatus = { faceDetected: false, ready: false, status: "chưa thấy mặt" };
    state.faceDrawingActive = false;
    updateCameraToolUI();
    liftPenDebounced();
    return;
  }
  const metrics = getFaceDrawingMetrics(landmarks);
  state.faceDrawingActive = Boolean(metrics.mouthOpen);
  state.faceStatus = {
    faceDetected: true,
    ready: metrics.mouthOpen,
    status: metrics.mouthOpen ? "đang vẽ" : "há miệng để vẽ",
  };
  updateCameraToolUI();
  drawFaceOverlay(ctx, landmarks, metrics);

  if (metrics.bothEyesClosed && Date.now() - state.faceLastClear > 1600) {
    state.faceLastClear = Date.now();
    clearDrawing();
    setStatus("Bút mặt: phát hiện chớp mắt, đã xóa nét.");
    return;
  }
  if (!state.running && !state.currentTarget) return;
  if (!metrics.mouthOpen) {
    updateFaceCenter(metrics.nose, false);
    liftPenDebounced();
    return;
  }
  state.penLiftFrames = 0;
  updateFaceCenter(metrics.nose, true);
  const point = smoothFacePoint(mapFacePointToCanvas(metrics.nose));
  if (!state.currentStroke) beginStroke(point);
  else extendStroke(point);
}

function getFaceDrawingMetrics(landmarks) {
  const p = (idx) => landmarks[idx] || landmarks[1];
  const dist = (a, b) => Math.hypot((a.x - b.x) * CANVAS_W, (a.y - b.y) * CANVAS_H);
  const nose = p(1);
  const faceHeight = Math.max(1, dist(p(10), p(152)));
  const mouthRatio = dist(p(13), p(14)) / faceHeight;
  const leftEyeRatio = dist(p(159), p(145)) / Math.max(1, dist(p(33), p(133)));
  const rightEyeRatio = dist(p(386), p(374)) / Math.max(1, dist(p(362), p(263)));
  return {
    nose,
    mouthRatio,
    mouthOpen: mouthRatio >= FACE_DRAW_MOUTH_THRESHOLD,
    leftEyeRatio,
    rightEyeRatio,
    bothEyesClosed: leftEyeRatio <= FACE_CLEAR_BLINK_THRESHOLD && rightEyeRatio <= FACE_CLEAR_BLINK_THRESHOLD,
  };
}

function updateFaceCenter(nose, drawing) {
  if (!state.faceCenter) {
    state.faceCenter = { x: nose.x, y: nose.y };
    return;
  }
  if (drawing) return;
  state.faceCenter.x += (nose.x - state.faceCenter.x) * FACE_CENTER_DRIFT_ALPHA;
  state.faceCenter.y += (nose.y - state.faceCenter.y) * FACE_CENTER_DRIFT_ALPHA;
}

function mapFacePointToCanvas(nose) {
  if (!state.faceCenter) state.faceCenter = { x: nose.x, y: nose.y };
  const x = CANVAS_W / 2 + (state.faceCenter.x - nose.x) * CANVAS_W * FACE_GAIN;
  const y = CANVAS_H / 2 + (nose.y - state.faceCenter.y) * CANVAS_H * FACE_GAIN;
  return {
    x: Math.max(8, Math.min(CANVAS_W - 8, x)),
    y: Math.max(8, Math.min(CANVAS_H - 8, y)),
    t: performance.now(),
    source: "face-nose",
  };
}

function ensureFaceEuro() {
  if (!state.faceEuroX) {
    // beta nhỏ hơn tay một chút vì cử động đầu chậm hơn -> mượt hơn.
    state.faceEuroX = makeOneEuro({ minCutoff: 1.0, beta: 0.01 });
    state.faceEuroY = makeOneEuro({ minCutoff: 1.0, beta: 0.01 });
  }
}
function resetFaceEuro() {
  if (state.faceEuroX) state.faceEuroX.reset();
  if (state.faceEuroY) state.faceEuroY.reset();
}
function smoothFacePoint(point) {
  ensureFaceEuro();
  return {
    x: state.faceEuroX.filter(point.x, point.t),
    y: state.faceEuroY.filter(point.y, point.t),
    t: point.t,
    source: "face-nose",
  };
}

function drawFaceOverlay(ctx, landmarks, metrics) {
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = metrics.mouthOpen ? "rgba(85, 230, 165, 0.95)" : "rgba(255, 207, 87, 0.95)";
  ctx.fillStyle = metrics.mouthOpen ? "rgba(85, 230, 165, 0.9)" : "rgba(255, 207, 87, 0.85)";
  const xy = (idx) => {
    const p = landmarks[idx];
    return { x: (1 - p.x) * CANVAS_W, y: p.y * CANVAS_H };
  };
  const chains = [
    [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33],
    [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263],
    [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61],
    [1, 2, 98, 327, 1],
  ];
  for (const chain of chains) {
    ctx.beginPath();
    chain.forEach((idx, i) => {
      if (!landmarks[idx]) return;
      const p = xy(idx);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
  }
  const cursor = mapFacePointToCanvas(metrics.nose);
  ctx.beginPath();
  ctx.arc(cursor.x, cursor.y, metrics.mouthOpen ? 13 : 9, 0, Math.PI * 2);
  ctx.fill();
  ctx.font = "700 15px system-ui";
  ctx.fillText(metrics.mouthOpen ? "VẼ" : "NHẤC", cursor.x + 14, cursor.y - 10);
  ctx.restore();
}

function onHandResults(results) {
  if (state.cameraTool !== "hand") return;
  const canvas = $("#handCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
  const landmarks = results.multiHandLandmarks?.[0];
  if (!landmarks) {
    // Mất tay 1-2 frame là chuyện thường của MediaPipe -> KHÔNG ngắt nét ngay,
    // chờ đủ số frame mới nhấc bút để nét không bị đứt khúc.
    liftPenDebounced();
    return;
  }
  drawHandSkeleton(ctx, landmarks);
  const wrist = landmarks[0];
  const index = landmarks[8];
  const indexPip = landmarks[6];
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  // Phát hiện "ngón trỏ duỗi để vẽ" KHÔNG phụ thuộc hướng tay (xoay ngang vẫn
  // đúng): so sánh khoảng cách đầu ngón -> cổ tay với khớp PIP -> cổ tay.
  //  - Ngón duỗi: đầu ngón ở xa cổ tay hơn PIP  -> ratio > 1.
  //  - Ngón gập (nhấc bút): đầu ngón co lại gần lòng bàn tay -> ratio nhỏ.
  // Hysteresis: bắt đầu vẽ cần duỗi rõ; ngừng khi gập rõ -> nét vừa liền vừa
  // tách bạch đúng lúc nhấc tay.
  const ratio = dist(index, wrist) / Math.max(1e-4, dist(indexPip, wrist));
  const DOWN_ON = 1.18;   // duỗi rõ mới bắt đầu nét mới
  const DOWN_OFF = 1.04;  // co lại quá mức này -> nhấc bút (tách nét)
  const wasDrawing = Boolean(state.currentStroke);
  const indexUp = wasDrawing ? ratio > DOWN_OFF : ratio > DOWN_ON;
  // Xòe cả bàn tay để xóa: 4 đầu ngón đều xa cổ tay (rotation-invariant).
  const openPalm = [8, 12, 16, 20].every(
    (tip) => dist(landmarks[tip], wrist) > dist(landmarks[tip - 2], wrist) * 1.1,
  );
  const now = Date.now();
  if (openPalm && now - state.lastPalmClear > 1600) {
    state.lastPalmClear = now;
    clearDrawing();
    setStatus("Đã xóa nét bằng thao tác xòe bàn tay.");
    return;
  }
  if (!state.running && !state.currentTarget) return;
  if (!indexUp) {
    // Vẫn hiển thị con trỏ (rỗng) để người dùng ngắm vị trí trước khi hạ bút vẽ.
    drawPenCursor(ctx, (1 - index.x) * CANVAS_W, index.y * CANVAS_H, false);
    state.lastRawHand = null;
    liftPenDebounced();
    return;
  }
  // Ngón trỏ đang vẽ -> reset bộ đếm nhấc bút
  state.penLiftFrames = 0;
  const raw = {
    x: (1 - index.x) * CANVAS_W,
    y: index.y * CANVAS_H,
    t: performance.now(),
  };
  // Tách nét trên toạ độ THÔ: nếu đầu ngón nhảy xa (nhấc tay đổi vị trí) thì
  // kết thúc nét cũ và reset bộ lọc -> nét mới bắt đầu sạch, không bị kéo nối.
  if (state.currentStroke && state.lastRawHand) {
    const jump = Math.hypot(raw.x - state.lastRawHand.x, raw.y - state.lastRawHand.y);
    if (jump > 150) {
      endStroke();
      resetHandEuro();
    }
  }
  state.lastRawHand = { x: raw.x, y: raw.y };
  const p = smoothPoint(raw);
  drawPenCursor(ctx, p.x, p.y, true);
  if (!state.currentStroke) beginStroke(p);
  else extendStroke(p);
}

// Vẽ con trỏ bút ở đầu ngón trỏ để người dùng biết CHÍNH XÁC điểm sẽ vẽ và
// trạng thái bút (đang vẽ = xanh đặc, đang nhấc = viền vàng rỗng).
function drawPenCursor(ctx, x, y, drawing) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, 9, 0, Math.PI * 2);
  if (drawing) {
    ctx.fillStyle = "rgba(85, 230, 165, 0.95)";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.stroke();
  } else {
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = "rgba(255, 207, 87, 0.95)";
    ctx.stroke();
  }
  ctx.restore();
}

// Chỉ thực sự kết thúc nét khi ngón trỏ hạ/mất tay LIÊN TỤC đủ lâu, tránh đứt
// nét do nhiễu nhận diện ở từng frame riêng lẻ.
const PEN_LIFT_FRAMES = 4;
function liftPenDebounced() {
  if (!state.currentStroke) return;
  state.penLiftFrames = (state.penLiftFrames || 0) + 1;
  if (state.penLiftFrames >= PEN_LIFT_FRAMES) {
    endStroke();
    state.penLiftFrames = 0;
    state.lastRawHand = null;
    if (state.cameraTool === "face") {
      state.faceFilteredPoint = null;
      resetFaceEuro();
    } else {
      state.filteredPoint = null;
      resetHandEuro();
    }
  }
}

// ---------------------------------------------------------------------------
// Bộ lọc One-Euro: tiêu chuẩn vàng cho vẽ bằng ngón tay/camera.
//  - Khi tay đứng yên/đi chậm: lọc mạnh -> hết rung, nét mượt.
//  - Khi tay di nhanh: giảm lọc -> bám sát, ít trễ -> vẽ đúng hình mong muốn.
// ---------------------------------------------------------------------------
function makeOneEuro({ minCutoff = 1.2, beta = 0.015, dCutoff = 1.0 } = {}) {
  let xPrev = null;
  let dxPrev = 0;
  let tPrev = null;
  const alpha = (cutoff, dt) => {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  };
  return {
    reset() { xPrev = null; dxPrev = 0; tPrev = null; },
    filter(x, t) {
      if (tPrev === null) { tPrev = t; xPrev = x; dxPrev = 0; return x; }
      let dt = (t - tPrev) / 1000;
      if (!(dt > 0)) dt = 1 / 60;
      tPrev = t;
      const dx = (x - xPrev) / dt;
      const aD = alpha(dCutoff, dt);
      const dxHat = aD * dx + (1 - aD) * dxPrev;
      dxPrev = dxHat;
      const cutoff = minCutoff + beta * Math.abs(dxHat);
      const a = alpha(cutoff, dt);
      const xHat = a * x + (1 - a) * xPrev;
      xPrev = xHat;
      return xHat;
    },
  };
}

function ensureHandEuro() {
  if (!state.euroX) {
    state.euroX = makeOneEuro({ minCutoff: 1.2, beta: 0.015 });
    state.euroY = makeOneEuro({ minCutoff: 1.2, beta: 0.015 });
  }
}
function resetHandEuro() {
  if (state.euroX) state.euroX.reset();
  if (state.euroY) state.euroY.reset();
}

// Làm mượt điểm ngón trỏ bằng One-Euro (thay cho EMA cũ).
function smoothPoint(point) {
  ensureHandEuro();
  return {
    x: state.euroX.filter(point.x, point.t),
    y: state.euroY.filter(point.y, point.t),
    t: point.t,
  };
}

function smoothPoint_legacy_unused(point) {
  if (!state.filteredPoint) {
    state.filteredPoint = point;
    return point;
  }
  const dx = point.x - state.filteredPoint.x;
  const dy = point.y - state.filteredPoint.y;
  const dist = Math.hypot(dx, dy);
  const alpha = Math.min(0.85, Math.max(0.3, dist / 90));
  state.filteredPoint = {
    x: state.filteredPoint.x + dx * alpha,
    y: state.filteredPoint.y + dy * alpha,
    t: point.t,
  };
  return state.filteredPoint;
}

function drawHandSkeleton(ctx, landmarks) {
  ctx.save();
  ctx.strokeStyle = "rgba(85, 230, 165, 0.95)";
  ctx.fillStyle = "rgba(87, 167, 255, 0.95)";
  ctx.lineWidth = 3;
  const lines = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
    [5, 9, 13, 17],
  ];
  for (const chain of lines) {
    ctx.beginPath();
    chain.forEach((idx, i) => {
      const p = landmarks[idx];
      const x = (1 - p.x) * CANVAS_W;
      const y = p.y * CANVAS_H;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc((1 - p.x) * CANVAS_W, p.y * CANVAS_H, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawGuide(label) {
  // Đã tắt lớp gợi ý vẽ hình trên canvas để người dùng tự vẽ từ đầu.
  const canvas = $("#guideCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
  return;
  if (!label) return;
  ctx.save();
  ctx.strokeStyle = "rgba(20, 40, 80, 0.34)";
  ctx.lineWidth = 10;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const x = (n) => n * CANVAS_W;
  const y = (n) => n * CANVAS_H;
  const circle = (cx, cy, r) => {
    ctx.beginPath();
    ctx.arc(x(cx), y(cy), r * Math.min(CANVAS_W, CANVAS_H), 0, Math.PI * 2);
    ctx.stroke();
  };
  const line = (a, b, c, d) => {
    ctx.beginPath();
    ctx.moveTo(x(a), y(b));
    ctx.lineTo(x(c), y(d));
    ctx.stroke();
  };
  const rect = (a, b, w, h) => ctx.strokeRect(x(a), y(b), x(w), y(h));
  switch (label) {
    case "apple":
      circle(0.5, 0.56, 0.22);
      line(0.5, 0.34, 0.5, 0.24);
      line(0.52, 0.27, 0.62, 0.22);
      break;
    case "pants":
      rect(0.35, 0.28, 0.3, 0.12);
      line(0.38, 0.4, 0.33, 0.78);
      line(0.5, 0.4, 0.47, 0.78);
      line(0.52, 0.4, 0.57, 0.78);
      line(0.65, 0.4, 0.68, 0.78);
      break;
    case "star":
      for (let i = 0; i < 5; i++) {
        const a = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
        const b = -Math.PI / 2 + (((i * 2 + 2) % 10) * Math.PI) / 5;
        line(
          0.5 + 0.25 * Math.cos(a),
          0.5 + 0.25 * Math.sin(a),
          0.5 + 0.25 * Math.cos(b),
          0.5 + 0.25 * Math.sin(b),
        );
      }
      break;
    case "square":
      rect(0.3, 0.25, 0.4, 0.5);
      break;
    case "book":
      // Gợi ý rõ hơn cho model: bìa chữ nhật + gáy giữa + 3 dòng trang bên trái.
      rect(0.25, 0.25, 0.5, 0.5);
      line(0.5, 0.25, 0.5, 0.75);
      line(0.32, 0.38, 0.46, 0.38);
      line(0.32, 0.48, 0.46, 0.48);
      line(0.32, 0.58, 0.46, 0.58);
      break;
    case "dog":
      circle(0.5, 0.52, 0.22);
      circle(0.35, 0.38, 0.08);
      circle(0.65, 0.38, 0.08);
      break;
    case "fish":
      circle(0.46, 0.52, 0.18);
      line(0.64, 0.52, 0.78, 0.38);
      line(0.64, 0.52, 0.78, 0.66);
      circle(0.39, 0.48, 0.025);
      break;
    default:
      circle(0.5, 0.5, 0.22);
      line(0.32, 0.72, 0.68, 0.72);
      break;
  }
  ctx.restore();
}

boot();
