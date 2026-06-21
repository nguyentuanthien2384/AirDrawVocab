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
  hands: null,
  handLoop: false,
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
  strokeModelAvailable: false,
};

const CANVAS_W = 960;
const CANVAS_H = 540;
const SUCCESS_THRESHOLD_BASE = 0.68;
const REALTIME_INTERVAL_MS = 620;

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
  const res = await fetch(url, { method: "POST", body: form, credentials: "same-origin" });
  if (!res.ok) throw new Error(await extractError(res));
  return res.json();
}

async function extractError(res) {
  try {
    const data = await res.json();
    return data.detail || data.error || `HTTP ${res.status}`;
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
  $("#registerBtn").addEventListener("click", () => loginOrRegister("register"));
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
  status.textContent = action === "login" ? "Đang đăng nhập..." : "Đang đăng ký...";
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
  try { await apiPostForm("/auth/logout", new FormData()); } catch {}
  state.user = null;
  renderLogin("Đã đăng xuất.");
}

async function loadGameData() {
  const data = await apiGet("/vocab");
  state.vocab = data.vocab || [];
  state.supportedVocab = state.vocab.filter((item) => item.recognition_supported);
  state.gamePool = state.supportedVocab.length ? state.supportedVocab : state.vocab;
  await refreshProfileAndLeaderboard();
}

async function refreshProfileAndLeaderboard() {
  try { state.profile = await apiGet("/game/profile"); } catch { state.profile = null; }
  try { state.leaderboard = (await apiGet("/game/leaderboard")).leaderboard || []; } catch { state.leaderboard = []; }
}

function renderGameShell() {
  const recognitionNote = state.supportedVocab.length < state.vocab.length
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
          </div>

          <div id="gameStage" class="game-stage mouse-mode">
            <video id="cameraVideo" autoplay muted playsinline></video>
            <canvas id="guideCanvas" width="${CANVAS_W}" height="${CANVAS_H}"></canvas>
            <canvas id="drawCanvas" width="${CANVAS_W}" height="${CANVAS_H}"></canvas>
            <canvas id="handCanvas" width="${CANVAS_W}" height="${CANVAS_H}"></canvas>
            <div class="stage-overlay">
              <div class="target-chip">Mục tiêu: <strong id="targetChip">---</strong></div>
              <div class="ai-chip">AI thấy: <strong id="aiChip">---</strong></div>
            </div>
            <div id="correctFlash" class="correct-flash">Correct +Score</div>
          </div>

          <div class="controls">
            <div class="control-row">
              <button id="startBtn" class="primary">Bắt đầu</button>
              <button id="clearBtn" class="secondary">Xóa nét</button>
              <button id="skipBtn" class="secondary" disabled>Bỏ qua</button>
              <button id="saveBtn" class="secondary" disabled>Lưu mẫu train</button>
              <span class="spacer"></span>
              <label class="range-wrap">AI Speed <input id="speedRange" type="range" min="450" max="1200" value="${REALTIME_INTERVAL_MS}" step="50"></label>
            </div>
            <div class="progress"><span id="confidenceProgress"></span></div>
            <p id="gameStatus" class="status">Bấm Bắt đầu để chơi. AI sẽ tự nhận diện trong lúc bạn vẽ.</p>
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
            <div class="side-header"><h3>Realtime AI</h3><span class="badge">Auto Judge</span></div>
            <div id="predictionList" class="prediction-list">AI chưa có dữ liệu.</div>
          </section>

          <section class="side-card">
            <div class="side-header"><h3>AI Judge Mode</h3><span id="gradeBadge" class="badge warn">---</span></div>
            <div class="judge-grid">
              <div class="judge-cell"><span>Shape</span><strong id="shapeScore">0</strong></div>
              <div class="judge-cell"><span>Clarity</span><strong id="clarityScore">0</strong></div>
              <div class="judge-cell"><span>Stroke</span><strong id="strokeScore">0</strong></div>
              <div class="judge-cell"><span>Speed</span><strong id="speedScore">0</strong></div>
            </div>
            <p id="judgeFeedback" class="feedback">AI sẽ đánh giá khi bạn bắt đầu vẽ.</p>
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
            <div class="pvp-controls three">
              <button id="exportDatasetBtn" class="secondary" type="button">Export data</button>
              <button id="trainStrokeBtn" class="secondary" type="button">Train stroke</button>
              <button id="trainImageBtn" class="secondary" type="button">Train image</button>
            </div>
            <div id="retrainBox" class="mini-list"><div class="mini-row"><span>Idle</span><small>Local/Colab pipeline</small></div></div>
          </section>
        </aside>
      </section>
    </main>
  `;

  wireGameEvents();
  setupCanvas();
  updateHud();
  renderProfile();
  renderLeaderboard();
  renderPvp();
  refreshRetrainStatus();
}

function wireGameEvents() {
  $("#logoutBtn").addEventListener("click", logout);
  $("#mouseModeBtn").addEventListener("click", () => setMode("mouse"));
  $("#cameraModeBtn").addEventListener("click", () => setMode("camera"));
  $("#startBtn").addEventListener("click", () => state.running ? endGame("Bạn đã kết thúc lượt chơi.") : startGame());
  $("#clearBtn").addEventListener("click", clearDrawing);
  $("#skipBtn").addEventListener("click", skipRound);
  $("#saveBtn").addEventListener("click", () => saveStrokeSample(false));
  $("#speakBtn").addEventListener("click", speakCurrentWord);
  $("#pvpBtn").addEventListener("click", togglePvp);
  $("#exportDatasetBtn").addEventListener("click", exportDataset);
  $("#trainStrokeBtn").addEventListener("click", () => startRetrain("stroke"));
  $("#trainImageBtn").addEventListener("click", () => startRetrain("image"));
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
  state.hasDrawn = true;
}

function extendStroke(point) {
  if (!state.lastPoint) {
    beginStroke(point);
    return;
  }
  drawSegment(state.lastPoint, point);
  state.currentStroke.push(point);
  state.lastPoint = point;
  state.hasDrawn = true;
}

function endStroke() {
  if (state.currentStroke && state.currentStroke.length > 0) {
    state.strokes.push(state.currentStroke);
  }
  state.currentStroke = null;
  state.lastPoint = null;
}

function drawSegment(a, b) {
  const ctx = $("#drawCanvas").getContext("2d");
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  const mx = (a.x + b.x) / 2;
  const my = (a.y + b.y) / 2;
  ctx.quadraticCurveTo(a.x, a.y, mx, my);
  ctx.stroke();
}

function clearDrawing() {
  const draw = $("#drawCanvas");
  draw.getContext("2d").clearRect(0, 0, CANVAS_W, CANVAS_H);
  const hand = $("#handCanvas");
  hand.getContext("2d").clearRect(0, 0, CANVAS_W, CANVAS_H);
  state.hasDrawn = false;
  state.currentStroke = null;
  state.strokes = [];
  state.lastPoint = null;
  state.filteredPoint = null;
  state.predictionBuffer = [];
  state.consecutiveCorrect = 0;
  state.prediction = null;
  state.judge = null;
  updatePredictionPanel(null);
  updateJudge(null);
  $("#aiChip").textContent = "---";
  $("#confidenceProgress").style.width = "0%";
}

async function setMode(mode) {
  state.mode = mode;
  const stage = $("#gameStage");
  stage.classList.toggle("camera-mode", mode === "camera");
  stage.classList.toggle("mouse-mode", mode === "mouse");
  $("#mouseModeBtn").classList.toggle("active", mode === "mouse");
  $("#cameraModeBtn").classList.toggle("active", mode === "camera");
  if (mode === "camera") {
    await startCamera();
    setStatus("Camera mode: dùng ngón trỏ để vẽ, mở cả bàn tay để xóa nhanh.", "ok");
  } else {
    stopCamera();
    setStatus("Mouse mode: vẽ trực tiếp trên khung trắng.");
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
  state.gamePool = shuffle(state.supportedVocab.length ? state.supportedVocab : state.vocab);
  $("#startBtn").textContent = "Kết thúc";
  $("#skipBtn").disabled = false;
  $("#saveBtn").disabled = false;
  nextRound();
  startGameTimer();
  startRealtimeAI();
  if (state.mode === "camera") await startCamera();
  setStatus("Final Boss Mode đang chạy: AI tự đoán real-time và tự qua màn khi đúng.", "ok");
}

function nextRound() {
  if (!state.running) return;
  if (state.level > state.gamePool.length) {
    endGame("Hoàn thành toàn bộ vòng chơi.");
    return;
  }
  clearDrawing();
  state.currentTarget = state.gamePool[(state.level - 1) % state.gamePool.length];
  state.roundTime = Math.max(35, 60 - Math.min(state.streak, 10) * 2);
  state.timeLeft = state.roundTime;
  state.roundStartedAt = Date.now();
  drawGuide(state.currentTarget.label);
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
  clearInterval(state.realtimeId);
  const speed = () => Number($("#speedRange")?.value || REALTIME_INTERVAL_MS);
  const tick = async () => {
    if (!state.running || !state.currentTarget || !state.hasDrawn || state.predictInFlight) return;
    await realtimePredict();
  };
  state.realtimeId = setInterval(tick, speed());
  $("#speedRange").addEventListener("change", () => {
    clearInterval(state.realtimeId);
    state.realtimeId = setInterval(tick, speed());
  });
}

async function realtimePredict() {
  state.predictInFlight = true;
  try {
    const blob = await captureDrawingBlob();
    const form = new FormData();
    form.append("file", blob, "drawing.png");
    form.append("target", state.currentTarget.label);
    form.append("source", state.mode === "camera" ? "camera" : "mouse");
    form.append("stroke_count", String(countStrokePoints()));
    form.append("elapsed_ms", String(Date.now() - state.roundStartedAt));
    let data = await apiPostForm("/predict_godmode", form);
    const strokeData = await tryStrokePrediction();
    if (strokeData?.available && strokeData.confidence > (data.confidence || 0)) {
      data = {
        ...data,
        label: strokeData.label,
        confidence: strokeData.confidence,
        confidence_percent: Math.round(strokeData.confidence * 10000) / 100,
        top5: strokeData.top5,
        is_correct: strokeData.is_correct,
        ai_source: "stroke-sequence",
      };
    } else {
      data.ai_source = "image-cnn";
    }
    state.prediction = data;
    state.judge = data.judge;
    updatePredictionPanel(data);
    updateJudge(data.judge);
    broadcastPvp({ type: "prediction", label: data.label, confidence: data.confidence, target: state.currentTarget?.label, score: state.score });
    handleRealtimeDecision(data);
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    state.predictInFlight = false;
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

function handleRealtimeDecision(data) {
  const conf = Number(data.confidence || 0);
  const target = state.currentTarget?.label;
  $("#aiChip").textContent = `${data.label} ${(conf * 100).toFixed(0)}%`;
  $("#confidenceProgress").style.width = `${Math.round(conf * 100)}%`;
  state.predictionBuffer.push({ label: data.label, confidence: conf });
  if (state.predictionBuffer.length > 6) state.predictionBuffer.shift();
  const threshold = Math.max(0.58, SUCCESS_THRESHOLD_BASE - Math.min(state.streak, 8) * 0.01);
  if (data.label === target && conf >= threshold) {
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
  broadcastPvp({ type: "score", score: state.score, target: state.currentTarget?.label, message: `Correct +${gained}` });
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
  const duration = Math.round((Date.now() - state.startedAt) / 1000);
  const accuracy = state.attempts ? (state.correct / state.attempts) * 100 : 0;
  const form = new FormData();
  form.append("score", String(state.score));
  form.append("level", String(state.level));
  form.append("streak", String(state.bestStreak));
  form.append("accuracy", String(accuracy.toFixed(1)));
  form.append("duration_seconds", String(duration));
  form.append("mode", state.mode);
  try { await apiPostForm("/game/session", form); } catch (err) { console.warn(err); }
  await refreshProfileAndLeaderboard();
  renderProfile();
  renderLeaderboard();
  setStatus(`${message} Score: ${state.score}. Accuracy: ${accuracy.toFixed(1)}%.`, "ok");
}

function stopAllLoops() {
  clearInterval(state.timerId);
  clearInterval(state.realtimeId);
  state.timerId = null;
  state.realtimeId = null;
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
  const S = (paths, extra = "") => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" fill="none" stroke="#55e6a5" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" ${extra}>${paths}</svg>`;
  const map = {
    apple:    S(`<circle cx="50" cy="58" r="26"/><path d="M50 32 Q55 20 65 18"/><path d="M50 34 Q44 24 36 26"/>`),
    baseball: S(`<circle cx="50" cy="50" r="28"/><path d="M34 26 Q38 38 34 50 Q30 62 34 74" stroke-width="3"/><path d="M66 26 Q62 38 66 50 Q70 62 66 74" stroke-width="3"/>`),
    book:     S(`<rect x="22" y="18" width="56" height="64" rx="3"/><line x1="50" y1="18" x2="50" y2="82"/><line x1="30" y1="32" x2="48" y2="32"/><line x1="30" y1="42" x2="48" y2="42"/><line x1="30" y1="52" x2="48" y2="52"/>`),
    bowtie:   S(`<polygon points="20,28 50,50 20,72" fill="rgba(85,230,165,0.15)"/><polygon points="80,28 50,50 80,72" fill="rgba(85,230,165,0.15)"/><circle cx="50" cy="50" r="6" fill="#55e6a5"/>`),
    diamond:  S(`<polygon points="50,14 84,50 50,86 16,50" fill="rgba(85,230,165,0.12)"/><line x1="16" y1="50" x2="50" y2="14"/><line x1="50" y1="14" x2="84" y2="50"/><line x1="84" y1="50" x2="50" y2="86"/><line x1="50" y1="86" x2="16" y2="50"/><line x1="16" y1="50" x2="84" y2="50"/>`),
    dog:      S(`<circle cx="50" cy="54" r="24"/><ellipse cx="30" cy="40" rx="10" ry="14" transform="rotate(-15,30,40)"/><ellipse cx="70" cy="40" rx="10" ry="14" transform="rotate(15,70,40)"/><circle cx="43" cy="50" r="3" fill="#55e6a5"/><circle cx="57" cy="50" r="3" fill="#55e6a5"/><ellipse cx="50" cy="60" rx="7" ry="4"/>`),
    door:     S(`<rect x="28" y="14" width="44" height="72" rx="2"/><circle cx="64" cy="52" r="3" fill="#55e6a5"/><path d="M28 86 H72"/>`),
    envelope: S(`<rect x="14" y="28" width="72" height="48" rx="3"/><path d="M14 28 L50 58 L86 28"/>`),
    eye:      S(`<ellipse cx="50" cy="50" rx="36" ry="22"/><circle cx="50" cy="50" r="12"/><circle cx="50" cy="50" r="6" fill="rgba(85,230,165,0.3)"/>`),
    fish:     S(`<ellipse cx="46" cy="50" rx="24" ry="16"/><polygon points="70,50 86,36 86,64" fill="rgba(85,230,165,0.15)"/><circle cx="36" cy="46" r="3" fill="#55e6a5"/><line x1="30" y1="50" x2="58" y2="40" stroke-width="2"/>`),
    hat:      S(`<path d="M20 70 Q50 60 80 70"/><path d="M35 70 Q36 38 50 34 Q64 38 65 70"/><rect x="15" y="68" width="70" height="8" rx="4"/>`),
    leaf:     S(`<path d="M50 82 Q22 60 24 34 Q36 18 50 18 Q64 18 76 34 Q78 60 50 82Z" fill="rgba(85,230,165,0.15)"/><line x1="50" y1="82" x2="50" y2="22"/><line x1="50" y1="42" x2="36" y2="56" stroke-width="2"/><line x1="50" y1="54" x2="38" y2="65" stroke-width="2"/><line x1="50" y1="42" x2="64" y2="56" stroke-width="2"/><line x1="50" y1="54" x2="62" y2="65" stroke-width="2"/>`),
    lightning: S(`<polygon points="58,12 36,52 52,52 42,88 72,44 54,44" fill="rgba(255,209,102,0.2)" stroke="#ffd166" stroke-width="3"/>`),
    moon:     S(`<path d="M70 28 Q86 50 70 72 Q50 82 34 72 Q54 68 58 50 Q54 32 34 28 Q50 18 70 28Z" fill="rgba(85,230,165,0.15)"/>`),
    pants:    S(`<rect x="28" y="18" width="44" height="20" rx="3"/><path d="M28 38 L28 82 L50 82 L50 56 L50 82 L72 82 L72 38"/>`),
    scissors: S(`<line x1="50" y1="50" x2="24" y2="22"/><line x1="50" y1="50" x2="76" y2="22"/><line x1="50" y1="50" x2="28" y2="80"/><line x1="50" y1="50" x2="72" y2="80"/><circle cx="30" cy="78" r="10"/><circle cx="70" cy="78" r="10"/>`),
    square:   S(`<rect x="18" y="18" width="64" height="64" rx="2"/>`),
    star:     S(`<polygon points="50,12 61,37 88,37 67,56 75,82 50,64 25,82 33,56 12,37 39,37" fill="rgba(85,230,165,0.15)"/>`),
    "t-shirt": S(`<path d="M20 22 L36 14 L50 24 L64 14 L80 22 L70 44 L60 40 L60 82 L40 82 L40 40 L30 44Z" fill="rgba(85,230,165,0.12)"/>`),
    cat:      S(`<circle cx="50" cy="56" r="22"/><polygon points="32,36 26,14 44,28" fill="rgba(85,230,165,0.2)"/><polygon points="68,36 74,14 56,28" fill="rgba(85,230,165,0.2)"/><circle cx="43" cy="52" r="3" fill="#55e6a5"/><circle cx="57" cy="52" r="3" fill="#55e6a5"/><path d="M42 62 Q50 67 58 62"/><line x1="30" y1="58" x2="50" y2="62" stroke-width="2"/><line x1="70" y1="58" x2="50" y2="62" stroke-width="2"/>`),
    sun:      S(`<circle cx="50" cy="50" r="18" fill="rgba(255,209,102,0.2)" stroke="#ffd166"/><line x1="50" y1="10" x2="50" y2="22" stroke="#ffd166"/><line x1="50" y1="78" x2="50" y2="90" stroke="#ffd166"/><line x1="10" y1="50" x2="22" y2="50" stroke="#ffd166"/><line x1="78" y1="50" x2="90" y2="50" stroke="#ffd166"/><line x1="22" y1="22" x2="30" y2="30" stroke="#ffd166"/><line x1="70" y1="70" x2="78" y2="78" stroke="#ffd166"/><line x1="78" y1="22" x2="70" y2="30" stroke="#ffd166"/><line x1="22" y1="78" x2="30" y2="70" stroke="#ffd166"/>`),
    tree:     S(`<ellipse cx="50" cy="36" rx="28" ry="24" fill="rgba(85,230,165,0.15)"/><rect x="44" y="58" width="12" height="26" rx="2"/>`),
    flower:   S(`<circle cx="50" cy="50" r="10" fill="rgba(255,209,102,0.3)" stroke="#ffd166"/><ellipse cx="50" cy="28" rx="9" ry="14" fill="rgba(85,230,165,0.2)"/><ellipse cx="50" cy="72" rx="9" ry="14" fill="rgba(85,230,165,0.2)"/><ellipse cx="28" cy="50" rx="14" ry="9" fill="rgba(85,230,165,0.2)"/><ellipse cx="72" cy="50" rx="14" ry="9" fill="rgba(85,230,165,0.2)"/><ellipse cx="34" cy="34" rx="9" ry="14" fill="rgba(85,230,165,0.2)" transform="rotate(-45,34,34)"/><ellipse cx="66" cy="34" rx="9" ry="14" fill="rgba(85,230,165,0.2)" transform="rotate(45,66,34)"/><ellipse cx="34" cy="66" rx="9" ry="14" fill="rgba(85,230,165,0.2)" transform="rotate(45,34,66)"/><ellipse cx="66" cy="66" rx="9" ry="14" fill="rgba(85,230,165,0.2)" transform="rotate(-45,66,66)"/>`),
    cloud:    S(`<ellipse cx="40" cy="58" rx="24" ry="18"/><ellipse cx="62" cy="58" rx="20" ry="16"/><ellipse cx="50" cy="46" rx="18" ry="16"/>`),
    umbrella: S(`<path d="M14 50 Q14 20 50 18 Q86 20 86 50Z"/><line x1="50" y1="18" x2="50" y2="76"/><path d="M50 76 Q50 86 40 86 Q30 86 30 76"/>`),
    key:      S(`<circle cx="36" cy="42" r="16"/><circle cx="36" cy="42" r="8"/><line x1="50" y1="42" x2="86" y2="42"/><line x1="74" y1="42" x2="74" y2="54"/><line x1="84" y1="42" x2="84" y2="52"/>`),
    cup:      S(`<path d="M26 30 L34 80 L66 80 L74 30Z"/><path d="M74 46 Q88 46 88 58 Q88 70 74 70"/>`),
    clock:    S(`<circle cx="50" cy="50" r="34"/><line x1="50" y1="50" x2="50" y2="24" stroke-width="5"/><line x1="50" y1="50" x2="68" y2="60" stroke-width="3"/><circle cx="50" cy="50" r="3" fill="#55e6a5"/>`),
    car:      S(`<rect x="12" y="44" width="76" height="30" rx="6"/><path d="M24 44 L32 24 L68 24 L76 44"/><circle cx="28" cy="76" r="10"/><circle cx="72" cy="76" r="10"/><rect x="36" y="28" width="14" height="14" rx="2"/><rect x="52" y="28" width="14" height="14" rx="2"/>`),
    bicycle:  S(`<circle cx="28" cy="62" r="20"/><circle cx="72" cy="62" r="20"/><line x1="28" y1="62" x2="50" y2="32"/><line x1="72" y1="62" x2="50" y2="32"/><line x1="50" y1="32" x2="28" y2="62"/><line x1="50" y1="32" x2="50" y2="48"/><line x1="44" y1="26" x2="56" y2="26"/>`),
    airplane: S(`<path d="M8 54 L50 36 L92 54 L74 54 L74 68 L50 62 L26 68 L26 54Z" fill="rgba(85,230,165,0.12)"/><path d="M50 36 L50 76"/><path d="M42 72 L58 72"/>`),
    house:    S(`<polygon points="50,14 88,46 78,46 78,86 22,86 22,46 12,46" fill="rgba(85,230,165,0.12)"/><rect x="38" y="62" width="24" height="24" rx="2"/>`),
    banana:   S(`<path d="M26 78 Q18 50 32 28 Q46 16 66 18 Q74 18 74 26 Q74 34 60 34 Q40 34 32 52 Q26 66 34 78Z" fill="rgba(255,209,102,0.2)" stroke="#ffd166"/>`),
    "ice cream": S(`<polygon points="50,88 26,42 74,42" fill="rgba(85,230,165,0.12)"/><circle cx="50" cy="34" r="18" fill="rgba(255,120,80,0.2)" stroke="#ff7850" stroke-width="3"/><path d="M38 28 Q44 22 50 28 Q56 22 62 28" stroke="#ff7850" stroke-width="2"/>`),
    cake:     S(`<rect x="20" y="52" width="60" height="34" rx="4" fill="rgba(85,230,165,0.12)"/><rect x="30" y="42" width="40" height="12" rx="2"/><line x1="34" y1="22" x2="34" y2="42"/><line x1="50" y1="18" x2="50" y2="42"/><line x1="66" y1="22" x2="66" y2="42"/><ellipse cx="34" cy="20" rx="4" ry="6" stroke="#ffd166"/><ellipse cx="50" cy="16" rx="4" ry="6" stroke="#ffd166"/><ellipse cx="66" cy="20" rx="4" ry="6" stroke="#ffd166"/>`),
    candle:   S(`<rect x="38" y="42" width="24" height="46" rx="3" fill="rgba(85,230,165,0.12)"/><path d="M50 12 Q54 22 50 30 Q46 22 50 12Z" fill="rgba(255,209,102,0.4)" stroke="#ffd166"/><line x1="50" y1="30" x2="50" y2="42"/>`),
    guitar:   S(`<path d="M50 16 Q58 16 60 24 L62 52 Q68 54 68 62 Q68 76 50 82 Q32 76 32 62 Q32 54 38 52 L40 24 Q42 16 50 16Z" fill="rgba(85,230,165,0.12)"/><circle cx="50" cy="62" r="8"/><line x1="42" y1="16" x2="58" y2="16"/>`),
    hammer:   S(`<line x1="42" y1="50" x2="70" y2="82"/><rect x="34" y="22" width="36" height="22" rx="4" fill="rgba(85,230,165,0.15)" transform="rotate(-30,52,33)"/>`),
    bed:      S(`<rect x="10" y="48" width="80" height="34" rx="4"/><rect x="10" y="44" width="18" height="38" rx="3"/><rect x="72" y="44" width="18" height="38" rx="3"/><ellipse cx="38" cy="48" rx="18" ry="10" fill="rgba(85,230,165,0.2)"/><ellipse cx="62" cy="48" rx="18" ry="10" fill="rgba(85,230,165,0.2)"/>`),
    chair:    S(`<rect x="28" y="34" width="44" height="8" rx="2"/><line x1="28" y1="42" x2="24" y2="82"/><line x1="72" y1="42" x2="76" y2="82"/><line x1="32" y1="42" x2="32" y2="82"/><line x1="68" y1="42" x2="68" y2="82"/><rect x="28" y="14" width="8" height="22" rx="2"/><rect x="64" y="14" width="8" height="22" rx="2"/><line x1="28" y1="24" x2="72" y2="24"/>`,),
  };
  return map[label] || S(`<text x="50" y="56" text-anchor="middle" font-size="36" fill="#55e6a5" stroke="none">${label.slice(0,2).toUpperCase()}</text>`);
}

function updateTargetPanel() {
  const item = state.currentTarget;
  if (!item) return;
  $("#targetWord").textContent = item.label;
  const svg = getIllustrationSVG(item.label);
  $("#wordMeta").innerHTML = `
    <div class="word-illustration" title="Hình minh hoạ: ${escapeHtml(item.label)}">${svg}</div>
    <p><b>Nghĩa:</b> ${escapeHtml(item.meaning_vi || item.label)}</p>
    <p><b>IPA:</b> ${escapeHtml(item.ipa || "")}</p>
    <p><b>Ví dụ:</b> ${escapeHtml(item.example_en || "")}</p>
    <p><b>Dịch:</b> ${escapeHtml(item.example_vi || "")}</p>
    <p><b>Gợi ý vẽ:</b> ${escapeHtml(item.drawing_hint || "Vẽ hình dạng chính rõ ràng.")}</p>
  `;
}

function updatePredictionPanel(data) {
  const box = $("#predictionList");
  if (!data || !data.top5) {
    box.textContent = "AI chưa có dữ liệu.";
    return;
  }
  const source = data.ai_source ? `<div class="mini-row"><span>AI source</span><small>${escapeHtml(data.ai_source)}</small></div>` : "";
  box.innerHTML = source + data.top5.map((p) => `
    <div class="pred-row">
      <div class="pred-label">${escapeHtml(p.label)}</div>
      <div class="pred-bar"><span style="width:${Math.round(p.confidence * 100)}%"></span></div>
      <div>${Math.round(p.confidence * 100)}%</div>
    </div>
  `).join("");
}

function updateJudge(judge) {
  $("#gradeBadge").textContent = judge?.grade || "---";
  $("#shapeScore").textContent = judge?.shape_score ?? 0;
  $("#clarityScore").textContent = judge?.clarity_score ?? 0;
  $("#strokeScore").textContent = judge?.stroke_score ?? 0;
  $("#speedScore").textContent = judge?.speed_score ?? 0;
  $("#judgeFeedback").textContent = judge?.feedback || "AI sẽ đánh giá khi bạn bắt đầu vẽ.";
}

function renderProfile() {
  const profile = state.profile;
  const stats = profile?.stats || {};
  $("#profileBox").innerHTML = `
    <div class="profile-item"><span>Games</span><strong>${stats.games || 0}</strong></div>
    <div class="profile-item"><span>Best score</span><strong>${stats.best_score || 0}</strong></div>
    <div class="profile-item"><span>Drawings</span><strong>${stats.drawings || 0}</strong></div>
    <div class="profile-item"><span>Accuracy</span><strong>${stats.accuracy || 0}%</strong></div>
  `;
  const weaknesses = profile?.weaknesses || [];
  $("#weaknessBox").innerHTML = weaknesses.length
    ? weaknesses.map((w) => `<div class="mini-row"><span>${escapeHtml(w.target)}</span><small>${w.accuracy}% / ${w.attempts} lần</small></div>`).join("")
    : `<div class="mini-row"><span>Chưa có dữ liệu yếu/mạnh</span><small>Chơi vài vòng để cập nhật</small></div>`;
}

function renderLeaderboard() {
  const rows = state.leaderboard || [];
  $("#leaderboardBox").innerHTML = rows.length
    ? rows.map((r, i) => `<div class="mini-row"><span>#${i + 1} ${escapeHtml(r.username || "guest")}</span><small>${r.score || 0} pts</small></div>`).join("")
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

function countStrokePoints() {
  return state.strokes.reduce((sum, stroke) => sum + stroke.length, 0) + (state.currentStroke?.length || 0);
}

async function saveStrokeSample(correct = false) {
  if (!state.currentTarget) return;
  const form = new FormData();
  form.append("target", state.currentTarget.label);
  form.append("predicted", state.prediction?.label || "");
  form.append("confidence", String(state.prediction?.confidence || 0));
  form.append("correct", correct ? "1" : "0");
  form.append("mode", state.mode);
  form.append("strokes_json", JSON.stringify(state.strokes));
  try { await apiPostForm("/game/stroke", form); } catch (err) { console.warn(err); }
  setStatus(correct ? "Đã lưu mẫu đúng vào dataset game." : "Đã lưu mẫu vẽ vào dataset game.", correct ? "ok" : "");
}


async function tryStrokePrediction() {
  if (!state.currentTarget || countStrokePoints() < 8) return null;
  try {
    const form = new FormData();
    form.append("strokes_json", JSON.stringify(state.strokes.concat(state.currentStroke ? [state.currentStroke] : [])));
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
      broadcastPvp({ type: "hello", score: state.score, target: state.currentTarget?.label || "" });
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
  if (!state.pvpConnected || !state.pvpSocket || state.pvpSocket.readyState !== WebSocket.OPEN) return;
  state.pvpSocket.send(JSON.stringify({ ...payload, score: state.score, level: state.level, ts: Date.now() }));
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
    ? players.map((p) => `<div class="mini-row"><span>${escapeHtml(p.username || "guest")}</span><small>${p.score || 0} pts</small></div>`).join("")
    : `<div class="mini-row"><span>${state.pvpConnected ? "Đang chờ người chơi" : "Chưa vào phòng"}</span><small>${escapeHtml(state.pvpRoom)}</small></div>`;
  const eventRows = state.pvpEvents.slice(0, 3).map((e) => `<div class="mini-row"><span>${escapeHtml(e.username || "system")}</span><small>${escapeHtml(e.message || e.label || e.type || "event")}</small></div>`).join("");
  box.innerHTML = playerRows + eventRows;
}

async function exportDataset() {
  const box = $("#retrainBox");
  try {
    const data = await apiGet("/dataset/export");
    box.innerHTML = `<div class="mini-row"><span>Export OK</span><small>${data.samples} samples</small></div><div class="mini-row"><span>Download</span><small>${data.download}</small></div>`;
  } catch (err) {
    box.innerHTML = `<div class="mini-row"><span>Export lỗi</span><small>${escapeHtml(err.message)}</small></div>`;
  }
}

async function startRetrain(mode) {
  const box = $("#retrainBox");
  const form = new FormData();
  form.append("mode", mode);
  form.append("epochs", mode === "stroke" ? "8" : "5");
  try {
    const data = await apiPostForm("/admin/retrain/start", form);
    box.innerHTML = `<div class="mini-row"><span>Retrain ${escapeHtml(mode)}</span><small>${data.ok ? "running" : "busy"}</small></div>`;
    setTimeout(refreshRetrainStatus, 1200);
  } catch (err) {
    box.innerHTML = `<div class="mini-row"><span>Retrain lỗi</span><small>${escapeHtml(err.message)}</small></div>`;
  }
}

async function refreshRetrainStatus() {
  const box = $("#retrainBox");
  if (!box) return;
  try {
    const data = await apiGet("/admin/retrain/status");
    state.retrainStatus = data;
    box.innerHTML = `<div class="mini-row"><span>${escapeHtml(data.status || "idle")}</span><small>${escapeHtml(data.mode || "-")}</small></div><div class="mini-row"><span>${escapeHtml(data.message || "")}</span><small>${data.process_running ? "running" : "stopped"}</small></div>`;
    if (data.process_running) setTimeout(refreshRetrainStatus, 2500);
  } catch (err) {
    box.innerHTML = `<div class="mini-row"><span>Status lỗi</span><small>${escapeHtml(err.message)}</small></div>`;
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

async function startCamera() {
  const video = $("#cameraVideo");
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Browser không hỗ trợ camera API.", "error");
    return;
  }
  if (!state.cameraStream) {
    try {
      state.cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 540 }, audio: false });
      video.srcObject = state.cameraStream;
      await video.play();
    } catch (err) {
      setStatus(`Không mở được camera: ${err.message}`, "error");
      return;
    }
  }
  if (!state.hands && window.Hands) {
    state.hands = new window.Hands({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
    state.hands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: 0.7, minTrackingConfidence: 0.65 });
    state.hands.onResults(onHandResults);
  }
  if (state.hands && !state.handLoop) {
    state.handLoop = true;
    handFrameLoop();
  } else if (!window.Hands) {
    setStatus("MediaPipe chưa tải được. Camera vẫn mở nhưng chưa tracking ngón tay.", "error");
  }
}

function stopCamera() {
  state.handLoop = false;
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((track) => track.stop());
    state.cameraStream = null;
  }
  const video = $("#cameraVideo");
  if (video) video.srcObject = null;
  state.filteredPoint = null;
}

async function handFrameLoop() {
  const video = $("#cameraVideo");
  while (state.handLoop && state.hands && video?.readyState >= 2) {
    try { await state.hands.send({ image: video }); } catch (err) { console.warn(err); }
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }
}

function onHandResults(results) {
  const canvas = $("#handCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
  const landmarks = results.multiHandLandmarks?.[0];
  if (!landmarks) {
    if (state.currentStroke) endStroke();
    return;
  }
  drawHandSkeleton(ctx, landmarks);
  const index = landmarks[8];
  const indexPip = landmarks[6];
  const indexUp = index.y < indexPip.y;
  const openPalm = [8, 12, 16, 20].every((tipIdx) => landmarks[tipIdx].y < landmarks[tipIdx - 2].y);
  const now = Date.now();
  if (openPalm && now - state.lastPalmClear > 1600) {
    state.lastPalmClear = now;
    clearDrawing();
    setStatus("Open palm: đã xóa nét.");
    return;
  }
  if (!state.running && !state.currentTarget) return;
  if (!indexUp) {
    if (state.currentStroke) endStroke();
    return;
  }
  const raw = { x: (1 - index.x) * CANVAS_W, y: index.y * CANVAS_H, t: performance.now() };
  const p = smoothPoint(raw);
  if (!state.currentStroke) beginStroke(p);
  else extendStroke(p);
}

function smoothPoint(point) {
  if (!state.filteredPoint) {
    state.filteredPoint = point;
    return point;
  }
  const alpha = 0.38;
  state.filteredPoint = {
    x: state.filteredPoint.x * (1 - alpha) + point.x * alpha,
    y: state.filteredPoint.y * (1 - alpha) + point.y * alpha,
    t: point.t,
  };
  return state.filteredPoint;
}

function drawHandSkeleton(ctx, landmarks) {
  ctx.save();
  ctx.strokeStyle = "rgba(85, 230, 165, 0.95)";
  ctx.fillStyle = "rgba(87, 167, 255, 0.95)";
  ctx.lineWidth = 3;
  const lines = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20],[5,9,13,17]];
  for (const chain of lines) {
    ctx.beginPath();
    chain.forEach((idx, i) => {
      const p = landmarks[idx];
      const x = (1 - p.x) * CANVAS_W;
      const y = p.y * CANVAS_H;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
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
  const canvas = $("#guideCanvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
  if (!label) return;
  ctx.save();
  ctx.strokeStyle = "rgba(20, 40, 80, 0.34)";
  ctx.lineWidth = 10;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const x = (n) => n * CANVAS_W;
  const y = (n) => n * CANVAS_H;
  const circle = (cx, cy, r) => { ctx.beginPath(); ctx.arc(x(cx), y(cy), r * Math.min(CANVAS_W, CANVAS_H), 0, Math.PI * 2); ctx.stroke(); };
  const line = (a,b,c,d) => { ctx.beginPath(); ctx.moveTo(x(a), y(b)); ctx.lineTo(x(c), y(d)); ctx.stroke(); };
  const rect = (a,b,w,h) => ctx.strokeRect(x(a), y(b), x(w), y(h));
  switch (label) {
    case "apple": circle(0.5,0.56,0.22); line(0.5,0.34,0.5,0.24); line(0.52,0.27,0.62,0.22); break;
    case "pants": rect(0.35,0.28,0.3,0.12); line(0.38,0.4,0.33,0.78); line(0.5,0.4,0.47,0.78); line(0.52,0.4,0.57,0.78); line(0.65,0.4,0.68,0.78); break;
    case "star": for(let i=0;i<5;i++){ const a=-Math.PI/2+i*2*Math.PI/5; const b=-Math.PI/2+((i*2+2)%10)*Math.PI/5; line(0.5+0.25*Math.cos(a),0.5+0.25*Math.sin(a),0.5+0.25*Math.cos(b),0.5+0.25*Math.sin(b)); } break;
    case "square": rect(0.3,0.25,0.4,0.5); break;
    case "book": rect(0.25,0.25,0.5,0.5); line(0.5,0.25,0.5,0.75); break;
    case "dog": circle(0.5,0.52,0.22); circle(0.35,0.38,0.08); circle(0.65,0.38,0.08); break;
    case "fish": circle(0.46,0.52,0.18); line(0.64,0.52,0.78,0.38); line(0.64,0.52,0.78,0.66); circle(0.39,0.48,0.025); break;
    default: circle(0.5,0.5,0.22); line(0.32,0.72,0.68,0.72); break;
  }
  ctx.restore();
}

boot();
