// Arm & Fist Tracker — 2-player
// Webcam + ml5.js bodyPose ("MoveNet" MultiPose) + p5.js
//
// Fullscreen camera feed with a subtle skeleton for the first two people on
// screen and a glowing fist marker on each wrist. Punches are detected from
// wrist motion, so the game reacts on the very first fast frame.

const SCORE_THRESHOLD = 0.3; // minimum confidence for a body keypoint

const CAM_W = 1280;
const CAM_H = 720;

let video;
let bodyPose;

let poses = [];
let handStates = []; // smoothed per-wrist data computed once per frame

let bodyConnections = [];

let videoReady = false;
let bodyReady = false;
let bodyStarted = false;

let mirror = true; // natural "mirror" view

// Intrinsic camera resolution (set once metadata loads). ml5 returns keypoints
// in this space, so we map them into screen space with the cover transform.
let videoW = CAM_W;
let videoH = CAM_H;

// Cover (object-fit: cover) transform: scales the camera to fill the whole
// window without distortion, cropping whatever overflows.
let coverScale = 1;
let coverOffX = 0;
let coverOffY = 0;

// Lightweight exponential smoothing to reduce keypoint jitter.
class Smooth {
  constructor(factor = 0.5) {
    this.factor = factor;
    this.map = new Map();
  }
  next(key, x, y, maxJump = 120) {
    const prev = this.map.get(key);
    if (!prev || Math.hypot(prev.x - x, prev.y - y) > maxJump) {
      this.map.set(key, { x, y });
      return { x, y };
    }
    const sx = prev.x + (x - prev.x) * this.factor;
    const sy = prev.y + (y - prev.y) * this.factor;
    this.map.set(key, { x: sx, y: sy });
    return { x: sx, y: sy };
  }
}

const bodySmooth = new Smooth(0.5);
const wristSmoothers = {}; // one smoother per player side + wrist
const wristMotion = {}; // last screen position / velocity per wrist

// Muñecos: historical art characters that pop out of the tree holes.
let munecos = [];
let nextSpawnAt = 0;

// Non-interactive characters that keep popping in and out behind the start
// screen, so the menu always feels alive.
let bgMunecos = [];
let bgNextSpawnAt = 0;

// Comic bursts shown where punches land.
let effects = [];

// Score per player (left / right of the divider), +10 for every character hit.
let scores = { L: 0, R: 0 };

// UI font, loaded from Google Fonts in index.html.
const UI_FONT = "Mochiy Pop One";

// --- Game state and music -------------------------------------------------

// "start" shows the menu, "roulette" spins the music pick, "playing" runs the match.
let gameState = "start";

// One fist from each player must be held up before the match starts.
const START_HOLD_MS = 700;
let startHoldMs = 0;
let startGate = { left: 0, right: 0, ready: false, progress: 0 };

// Music roulette: after the fist gate the glow hops between the three tracks
// with slowing steps and lands on a random winner.
const ROULETTE_STEPS = 14;
let rouletteStartAt = 0;
let rouletteStep = 0;
let rouletteIndex = 0;
let rouletteWinner = 0;
let rouletteLandedAt = 0;
let rouletteSchedule = [];

// Three background tracks by AudioCoffee (CC BY-NC 4.0). The first one is also
// the default src in index.html so it preloads with the page.
const TRACKS = [
  {
    name: "Winter Fun",
    url: "https://cdn.freesound.org/previews/717/717284_15232790-hq.mp3",
  },
  {
    name: "Uplifting Kiddy",
    url: "https://cdn.freesound.org/previews/815/815019_15232790-hq.mp3",
  },
  {
    name: "Dreamscape",
    url: "https://cdn.freesound.org/previews/848/848374_15232790-hq.mp3",
  },
];
let bgmEl = null;

// The 3 character portraits (Kandinsky, Shakespeare, Van Gogh), loaded async.
const CHAR_FILES = ["kandinsky.svg", "shakespeare.svg", "vangogh.svg"];
const charImgs = [];

// Width/height ratio of a character (all three share the same viewBox).
function charAspect(type) {
  const img = charImgs[type];
  return img && img.height ? img.width / img.height : 0.793;
}

let ambientImg;
let imgReady = false;

// Comic burst shown where a punch lands (PARTICULAS-02.svg, copied as particulas.svg).
let burstImg;

// Boxing glove drawn in front of each tracked fist. The SVG holds a pair, so
// each hand draws only its own half (crop fractions measured from the viewBox).
let gloveImg;
let gloveReady = false;
const GLOVE_CROP = {
  left: { sx: 0.0, sy: 0.0032, sw: 0.4614, sh: 0.9968 },
  right: { sx: 0.5451, sy: 0.0043, sw: 0.4549, sh: 0.9957 },
};

// The tree holes (normalized to the SVG viewBox 2573.89 x 1562.94) that the
// muñecos pop out of. Centers measured from the dark knothole paths in the art:
// 2 left, 1 right-upper, and 2 at the bottom.
const MUNECO_HOLES = [
  { x: 0.02343, y: 0.37366 }, // left-upper
  { x: 0.08561, y: 0.55773 }, // left-lower
  { x: 0.96657, y: 0.35916 }, // right-upper
  { x: 0.29716, y: 0.79254 }, // bottom-left
  { x: 0.66974, y: 0.83954 }, // bottom-right
];

// Rotation for a hole based on its side: left = +90°, right = -90°, bottom = 0°.
function holeRotation(x) {
  if (x < 0.2) return HALF_PI; // left side, clockwise
  if (x > 0.8) return -HALF_PI; // right side, counterclockwise
  return 0; // bottom, head up
}

function setup() {
  const cnv = createCanvas(windowWidth, windowHeight);
  cnv.parent("app");

  // Preload the UI font so the score uses it as soon as it is available.
  if (document.fonts && document.fonts.load) {
    document.fonts.load(`16px "${UI_FONT}"`).catch(() => {});
  }

  // Load the nature frame image asynchronously; draw it once it's ready.
  loadImage("ambiente.svg", onImageLoaded, onImageError);

  // Load the 3 character SVGs.
  CHAR_FILES.forEach((f, i) => {
    loadImage(f, (img) => { charImgs[i] = img; }, () => console.warn("[tracker] no cargó " + f));
  });

  // Load the punch burst.
  loadImage(
    "particulas.svg",
    (img) => { burstImg = img; },
    () => console.warn("[tracker] no cargó particulas.svg")
  );

  // Load the boxing glove that covers each tracked fist.
  loadImage(
    "guante.svg",
    (img) => { gloveImg = img; gloveReady = true; },
    () => console.warn("[tracker] no cargó guante.svg")
  );

  // Preload the punch sound (a small pool so hits can overlap).
  initHitSound();

  // Grab the shared background-music element (its src is set by index.html).
  initMusic();

  // ml5 1.4 returns a Promise that resolves to the model, and only the model
  // has detectStart(). Older builds return the model directly, so handle both.
  loadPoseModel(ml5.bodyPose("MoveNet", { modelType: "MULTIPOSE_LIGHTNING" }), "bodyPose", onBodyPoseReady);

  initCamera();

  nextSpawnAt = millis() + randomSpawnDelay();
  bgNextSpawnAt = millis() + 350;

  // Diagnostics: warn if the model never becomes ready (network/CORS issue).
  window.setTimeout(() => {
    if (!bodyReady) console.warn("[tracker] bodyPose no cargó en 15s (¿red/TFHub bloqueado?)");
  }, 15000);
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  updateCoverTransform();
}

// --- Camera ---------------------------------------------------------------

function initCamera() {
  if (bodyStarted) { bodyPose.detectStop(); bodyStarted = false; }
  videoReady = false;
  if (video) {
    if (video.elt && video.elt.srcObject) {
      video.elt.srcObject.getTracks().forEach((t) => t.stop());
    }
    video.remove();
  }

  video = createCapture(
    { video: { width: CAM_W, height: CAM_H, facingMode: "user" }, audio: false },
    onVideoReady
  );
  video.hide();
}

function updateCoverTransform() {
  const va = videoW / videoH;
  const ca = width / height;
  if (ca > va) {
    coverScale = height / videoH;
    coverOffX = (width - videoW * coverScale) / 2;
    coverOffY = 0;
  } else {
    coverScale = width / videoW;
    coverOffX = 0;
    coverOffY = (height - videoH * coverScale) / 2;
  }
}

function onImageLoaded(img) {
  ambientImg = img; // p5 2.x delivers the real image via the callback
  imgReady = true;
  console.log("[tracker] ambiente cargado", img.width, "x", img.height);
}

function onImageError() {
  console.warn("[tracker] no se pudo cargar ambiente.svg");
}

// --- Readiness (body pose drives the hands) -------------------------------

function onVideoReady() {
  videoW = video.elt.videoWidth || CAM_W;
  videoH = video.elt.videoHeight || CAM_H;
  updateCoverTransform();
  videoReady = true;
  tryStartBody();
}

// ml5 1.x can return either the model or a Promise for it. Normalize both so the
// readiness callbacks always receive a real model with detectStart().
function loadPoseModel(result, name, onReady) {
  if (result && typeof result.then === "function") {
    result
      .then((model) => onReady(model))
      .catch((err) => console.error(`[tracker] ${name} no cargó:`, err));
  } else {
    onReady(result);
  }
}

function onBodyPoseReady(model) {
  if (model) bodyPose = model;
  bodyReady = true;
  console.log("[tracker] bodyPose listo");
  tryStartBody();
}

function tryStartBody() {
  if (bodyStarted || !videoReady || !bodyReady) return;
  bodyStarted = true;
  bodyPose.detectStart(video, gotPoses);
  bodyConnections = bodyPose.getConnections() || [];
  console.log("[tracker] bodyPose detectStart iniciado");
}

function gotPoses(results) {
  poses = (results || []).slice(0, 2); // first two people only
  for (const pose of poses) {
    for (const kp of pose.keypoints) {
      kp.x = kp.x * coverScale + coverOffX;
      kp.y = kp.y * coverScale + coverOffY;
    }
  }
}

// --- Main loop ------------------------------------------------------------

function draw() {
  background(0);

  if (gameState !== "playing") {
    drawStartScreen();
    return;
  }

  drawCameraFeed();
  if (videoReady && bodyStarted) drawBodies();

  // Hands are smoothed once per frame, then punches are resolved against the
  // muñecos before drawing so a hit character vanishes immediately.
  if (videoReady && bodyStarted) updateHands();

  updateMunecos();
  if (videoReady && bodyStarted) resolveFistHits();
  updateEffects();

  if (videoReady && bodyStarted) drawHands();

  // Nature frame first, then the muñecos on top so the trunk never covers them.
  if (imgReady && ambientImg) {
    image(ambientImg, 0, 0, width, height);
  }

  drawMunecos();

  // Impact feedback stays visible over the frame.
  drawEffects();

  if (videoReady) drawDivider();

  drawScores();

  drawStatusDot();
}

// --- Camera feed (shared by the start screen and the match) ----------------

function drawCameraFeed(pad = 0) {
  if (!videoReady) return;
  push();
  if (mirror) {
    translate(width, 0);
    scale(-1, 1);
  }
  image(
    video,
    coverOffX - pad,
    coverOffY - pad,
    videoW * coverScale + pad * 2,
    videoH * coverScale + pad * 2
  );
  pop();
}

// Canvas 2D blur. Kept in one place so it can be tuned or disabled easily.
function setBlur(px) {
  const ctx = drawingContext;
  if (!ctx || !("filter" in ctx)) return;
  ctx.filter = px > 0 ? `blur(${px}px)` : "none";
}

// --- Start screen ----------------------------------------------------------

function drawStartScreen() {
  updateBgMunecos();

  // Blurred backdrop: camera + frame + the background cast, so the menu reads
  // as a paused-but-alive version of the scene.
  const blur = Math.max(4, width * 0.008);
  const bleed = blur * 3;
  setBlur(blur);

  drawCameraFeed(bleed);

  if (imgReady && ambientImg) {
    image(ambientImg, -bleed, -bleed, width + bleed * 2, height + bleed * 2);
  }

  for (const m of bgMunecos) drawMuneco(m);

  setBlur(0);

  // Gentle dark veil over the blurred scene.
  noStroke();
  fill(10, 6, 16, 62);
  rect(0, 0, width, height);

  // Hands are smoothed once per frame; the same states feed the start gate.
  if (videoReady && bodyStarted) {
    updateHands();
    drawHands();
  }

  const layout = startLayout();

  if (gameState === "roulette") {
    updateRoulette();
    if (gameState !== "roulette") return; // landed: the match already began
  } else {
    updateStartGate();
  }

  drawStartTitle(layout);

  // Two short lines so it also fits on narrow screens.
  drawMenuText("UN PUÑO DE CADA JUGADOR", width / 2, layout.line1Y, layout.lineSize, [255, 246, 226]);
  drawMenuText("PARA EMPEZAR", width / 2, layout.line2Y, layout.lineSize, [255, 246, 226]);

  // Either the fist indicators (when tracking works) or a short status line.
  if (videoReady && bodyStarted) {
    drawFistDots(layout);
  } else {
    drawCameraStatus(layout.dotsY);
  }

  drawTrackPicker(layout);
}

// Vertical rhythm of the menu: every gap is derived from a text size, so the
// blocks keep the same breathing room on any screen.
function startLayout() {
  const lineSize = Math.max(12, width * 0.023);
  const labelSize = Math.max(10, width * 0.016);
  const titleSize = Math.max(42, width * 0.112);

  const titleY = height * 0.235;
  const line1Y = height * 0.365;
  const line2Y = line1Y + lineSize * 1.7;
  const dotsY = line2Y + lineSize * 3.2;

  const ph = Math.max(34, Math.min(height * 0.052, 58));
  const gap = ph * 0.32;
  const pw = Math.min(width * 0.56, 460);
  const total = ph * 3 + gap * 2;
  const y0 = Math.min(height * 0.605, height - total - 20);
  const pills = TRACKS.map((_, i) => ({ x: (width - pw) / 2, y: y0 + i * (ph + gap), w: pw, h: ph }));

  return {
    titleSize,
    titleY,
    lineSize,
    line1Y,
    line2Y,
    dotsY,
    labelSize,
    labelY: y0 - labelSize * 2.3,
    pills,
  };
}

function drawStartTitle(layout) {
  push();
  textAlign(CENTER, CENTER);
  textFont(UI_FONT);
  textSize(layout.titleSize);

  // Soft shadow.
  noStroke();
  fill(16, 7, 14, 130);
  text("EMPEZAR", width / 2 + 3, layout.titleY + 5);

  // Same artwork as Empezar_Texto-03.svg: Mochiy Pop One in #8a336a, with a
  // cream edge so it stays readable over the live camera.
  stroke(255, 243, 210);
  strokeWeight(Math.max(5, width * 0.0095));
  fill(138, 51, 106);
  text("EMPEZAR", width / 2, layout.titleY);
  pop();
}

function drawMenuText(str, x, y, size, rgb) {
  push();
  textAlign(CENTER, CENTER);
  textFont(UI_FONT);
  textSize(size);
  noStroke();
  fill(16, 7, 14, 140);
  text(str, x + 1.5, y + 2);
  fill(rgb[0], rgb[1], rgb[2]);
  text(str, x, y);
  pop();
}

// Reads the tracked wrists and decides when both players have a fist up.
function updateStartGate() {
  const up = handStates.filter((hs) => hs.raised);
  const left = up.filter((hs) => hs.side === "L").length;
  const right = up.filter((hs) => hs.side === "R").length;
  const ready = left >= 1 && right >= 1;
  const dt = Math.min(deltaTime, 100);

  if (ready) startHoldMs += dt;
  else startHoldMs = Math.max(0, startHoldMs - dt * 2);

  if (startHoldMs >= START_HOLD_MS) startRoulette();

  startGate = {
    left,
    right,
    ready,
    progress: constrain(startHoldMs / START_HOLD_MS, 0, 1),
  };
  return startGate;
}

function drawFistDots(layout) {
  const cx = width / 2;
  const cy = layout.dotsY;
  const r = Math.max(8, width * 0.012);
  const gap = r * 3.8;
  const xs = [cx - gap / 2, cx + gap / 2];

  const lit = [startGate.left >= 1, startGate.right >= 1];
  const cols = [
    [255, 211, 110], // left player: gold
    [168, 230, 207], // right player: mint
  ];

  for (let i = 0; i < 2; i++) {
    if (lit[i]) {
      noStroke();
      if (drawingContext) {
        drawingContext.shadowBlur = 12;
        drawingContext.shadowColor = `rgba(${cols[i][0]}, ${cols[i][1]}, ${cols[i][2]}, 0.9)`;
      }
      fill(cols[i][0], cols[i][1], cols[i][2]);
      circle(xs[i], cy, r * 2);
      if (drawingContext) drawingContext.shadowBlur = 0;
    } else {
      noFill();
      strokeWeight(2);
      stroke(255, 245, 220, 155);
      circle(xs[i], cy, r * 2);
    }
  }

  // Fill bar while both fists are held.
  if (startGate.progress > 0) {
    const bw = gap + r * 3.4;
    const bh = Math.max(4, height * 0.006);
    const by = cy + r * 2.2;
    noStroke();
    fill(255, 245, 220, 50);
    rect(cx - bw / 2, by, bw, bh, bh / 2);
    fill(255, 226, 138, 240);
    rect(cx - bw / 2, by, bw * startGate.progress, bh, bh / 2);
  }
}

function drawCameraStatus(y) {
  let msg = null;
  if (!videoReady) msg = "ESPERANDO CÁMARA…";
  else if (!bodyStarted) msg = "CARGANDO DETECCIÓN…";
  if (!msg) return;
  drawMenuText(msg, width / 2, y, Math.max(10, width * 0.017), [255, 236, 200]);
}

function drawTrackPicker(layout) {
  const spinning = gameState === "roulette";
  const landed = spinning && rouletteStep >= rouletteSchedule.length;
  const active = spinning ? rouletteIndex : -1;

  let label = "LA MÚSICA SE ELIGE AL AZAR";
  if (spinning) label = landed ? "¡MÚSICA ELEGIDA!" : "ELIGIENDO MÚSICA…";
  drawMenuText(label, width / 2, layout.labelY, layout.labelSize, [255, 240, 205]);

  for (let i = 0; i < layout.pills.length; i++) {
    const r = layout.pills[i];
    const isActive = i === active;
    const isWinner = landed && i === rouletteWinner;

    push();
    strokeWeight(2);

    if (isActive || isWinner) {
      // Warm glow, pulsing while the roulette is still running.
      if (drawingContext) {
        const pulse = 0.5 + 0.5 * Math.sin(millis() / 85);
        drawingContext.shadowBlur = isWinner ? 34 : 16 + pulse * 20;
        drawingContext.shadowColor = "rgba(255, 205, 96, 0.95)";
      }
      fill(255, 211, 110, 245);
      stroke(255, 252, 240, 240);
    } else {
      fill(20, 12, 20, 148);
      stroke(255, 245, 220, 110);
    }

    rect(r.x, r.y, r.w, r.h, r.h / 2);
    if (drawingContext) drawingContext.shadowBlur = 0;

    noStroke();
    if (isActive || isWinner) fill(43, 26, 16);
    else fill(255, 245, 220);
    textAlign(CENTER, CENTER);
    textFont(UI_FONT);
    textSize(Math.max(9, Math.min(r.h * 0.32, r.w * 0.082)));
    text(TRACKS[i].name.toUpperCase(), r.x + r.w / 2, r.y + r.h / 2 + 1);
    pop();
  }
}

// Builds a random hop schedule that slows down towards the end and always
// finishes on the pre-picked winner.
function startRoulette() {
  gameState = "roulette";
  startHoldMs = 0;
  rouletteWinner = Math.floor(random(TRACKS.length));
  rouletteStep = 0;
  rouletteIndex = 0;
  rouletteLandedAt = 0;
  rouletteStartAt = millis();
  rouletteSchedule = [];

  let t = 0;
  let prev = 0;
  for (let i = 0; i < ROULETTE_STEPS; i++) {
    const k = i / (ROULETTE_STEPS - 1);
    t += lerp(70, 330, k * k * k); // longer waits as it decelerates

    let idx;
    if (i === ROULETTE_STEPS - 1) {
      idx = rouletteWinner;
    } else {
      idx = Math.floor(random(TRACKS.length));
      // Never repeat the same pill twice in a row.
      if (idx === prev) idx = (idx + 1 + Math.floor(random(TRACKS.length - 1))) % TRACKS.length;
    }

    prev = idx;
    rouletteSchedule.push({ at: t, index: idx });
  }
}

function updateRoulette() {
  const now = millis();
  const elapsed = now - rouletteStartAt;

  while (rouletteStep < rouletteSchedule.length && elapsed >= rouletteSchedule[rouletteStep].at) {
    rouletteIndex = rouletteSchedule[rouletteStep].index;
    rouletteStep++;
  }

  if (rouletteStep >= rouletteSchedule.length) {
    if (!rouletteLandedAt) rouletteLandedAt = now;
    if (now - rouletteLandedAt >= 900) {
      selectTrack(rouletteWinner);
      beginMatch();
    }
  }
}

function beginMatch() {
  gameState = "playing";
  scores.L = 0;
  scores.R = 0;
  munecos.length = 0;
  effects.length = 0;
  bgMunecos.length = 0;
  startHoldMs = 0;
  for (const key in lastFistHitAt) delete lastFistHitAt[key];
  nextSpawnAt = millis() + 700;
  bgNextSpawnAt = millis() + 350;
}

// --- Music ----------------------------------------------------------------

function initMusic() {
  bgmEl = document.getElementById("bgm");

  // Warm the cache for the other tracks so the roulette winner starts fast.
  for (let i = 1; i < TRACKS.length; i++) {
    const a = new Audio();
    a.preload = "auto";
    a.src = TRACKS[i].url;
  }
}

function selectTrack(i) {
  if (i < 0 || i >= TRACKS.length) return;
  if (!bgmEl) return;
  bgmEl.src = TRACKS[i].url;
  bgmEl.load();
  playMusic();
}

function playMusic() {
  if (!bgmEl) return;
  const p = bgmEl.play();
  if (p && p.catch) p.catch(() => {});
}

// --- Score -----------------------------------------------------------------

function drawScores() {
  const size = Math.max(26, width * 0.045);
  const margin = Math.max(14, width * 0.035);

  push();
  textFont(UI_FONT);
  textSize(size);
  textAlign(CENTER, CENTER);

  // Left player (warm gold) and right player (cool mint).
  drawScore(scores.L, margin + size * 0.6, margin + size * 0.55, [255, 211, 110]);
  drawScore(scores.R, width - margin - size * 0.6, margin + size * 0.55, [168, 230, 207]);

  pop();
}

function drawScore(value, x, y, rgb) {
  const weight = Math.max(4, width * 0.006);

  // Soft drop shadow.
  noStroke();
  fill(0, 0, 0, 90);
  text(value, x + 3, y + 4);

  // Thick dark outline so the number reads on top of the foliage.
  stroke(43, 26, 16);
  strokeWeight(weight);
  fill(rgb[0], rgb[1], rgb[2]);
  text(value, x, y);

  noStroke();
}

// --- Body (subtle skeleton, up to 2 people) -------------------------------

function drawBodies() {
  poses.forEach((pose, i) => drawPose(pose, i));
}

function drawPose(pose, index) {
  const sm = [];
  const byName = {};
  for (const kp of pose.keypoints) {
    // Per-person smoothing key so the two players don't share state.
    const p = bodySmooth.next(`${index}:${kp.name}`, kp.x, kp.y);
    const s = { ...kp, x: p.x, y: p.y };
    sm.push(s);
    byName[s.name] = s;
  }

  // Subtle full-body skeleton.
  stroke(255, 255, 255, 110);
  strokeWeight(2);
  for (const [a, b] of bodyConnections) {
    const pa = byName[a];
    const pb = byName[b];
    if (pa && pb && kpConfidence(pa) > SCORE_THRESHOLD && kpConfidence(pb) > SCORE_THRESHOLD) {
      line(px(pa.x), pa.y, px(pb.x), pb.y);
    }
  }

  // Subtle joints.
  noStroke();
  fill(255, 255, 255, 130);
  for (const kp of sm) {
    if (kpConfidence(kp) > SCORE_THRESHOLD) circle(px(kp.x), kp.y, 5);
  }
}

// Subtle vertical divider at the center of the screen.
function drawDivider() {
  stroke(255, 255, 255, 110);
  strokeWeight(2);
  const dash = 12;
  const gap = 16;
  for (let y = 0; y < height; y += dash + gap) {
    line(width / 2, y, width / 2, Math.min(y + dash, height));
  }
}

// Tiny non-textual status dot: green = tracking someone, amber = waiting.
function drawStatusDot() {
  noStroke();
  if (poses.length > 0) fill(46, 224, 106, 200);
  else fill(255, 176, 32, 200);
  circle(14, 14, 8);
}

// --- Hands (wrists from body pose) ----------------------------------------

// A hand counts as a punch when the wrist moves faster than this (screen px
// per second) and is inside a character. Resting or drifting stays below it.
const PUNCH_MIN_SPEED_RATIO = 0.65; // of the screen height per second
const PUNCH_MEMORY_MS = 170; // a fast frame keeps the fist "hot" this long
const GLOVE_SHOW_MS = 420; // the glove flashes in after a punch, then fades

function updateHands() {
  const now = millis();
  const punchSpeed = Math.max(420, height * PUNCH_MIN_SPEED_RATIO);
  const next = [];

  for (const pose of poses) {
    const byName = {};
    for (const kp of pose.keypoints) {
      if (kpConfidence(kp) > SCORE_THRESHOLD) byName[kp.name] = kp;
    }

    const side = poseSide(byName);
    if (!side) continue;

    for (const wristName of ["left_wrist", "right_wrist"]) {
      const wrist = byName[wristName];
      if (!wrist) continue;

      const key = `${side}:${wristName}`;
      if (!wristSmoothers[key]) wristSmoothers[key] = new Smooth(0.7);

      // Body-pose wrists are noisier than hand landmarks, so smooth them once
      // here and reuse the same value for drawing and hit detection.
      const s = wristSmoothers[key].next("wrist", wrist.x, wrist.y, 90);
      const x = px(s.x);
      const y = s.y;

      let st = wristMotion[key];
      if (!st) {
        st = wristMotion[key] = { x, y, prevX: x, prevY: y, at: now, speed: 0, angle: 0, punchUntil: 0, gloveUntil: 0 };
      }

      const dt = (now - st.at) / 1000;
      if (dt >= 0.008) {
        const vx = (x - st.x) / dt;
        const vy = (y - st.y) / dt;
        st.speed = st.speed * 0.3 + Math.hypot(vx, vy) * 0.7;
        st.prevX = st.x;
        st.prevY = st.y;
        st.x = x;
        st.y = y;
        st.at = now;
      }

      if (st.speed >= punchSpeed) {
        st.punchUntil = now + PUNCH_MEMORY_MS;
        st.gloveUntil = now + GLOVE_SHOW_MS;
      }

      const shoulder = byName[wristName === "left_wrist" ? "left_shoulder" : "right_shoulder"];
      const raised = !!shoulder && y < shoulder.y - height * 0.04;

      // Point the glove along the forearm so a raised fist reads naturally.
      const elbow = byName[wristName === "left_wrist" ? "left_elbow" : "right_elbow"];
      let targetAngle = 0;
      if (elbow) targetAngle = constrain(Math.atan2(x - px(elbow.x), -(y - elbow.y)), -1.2, 1.2);
      st.angle += (targetAngle - st.angle) * 0.25;

      next.push({
        key,
        side,
        limb: wristName === "left_wrist" ? "left" : "right",
        x,
        y,
        prevX: st.prevX,
        prevY: st.prevY,
        speed: st.speed,
        angle: st.angle,
        punchUntil: st.punchUntil,
        gloveUntil: st.gloveUntil,
        raised,
      });
    }
  }

  handStates = next;
}

// A player owns the side of the screen their torso center is on; more stable
// than testing each wrist independently.
function poseSide(byName) {
  let sum = 0;
  let n = 0;
  for (const name of ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]) {
    const kp = byName[name];
    if (!kp) continue;
    sum += px(kp.x);
    n++;
  }
  if (!n) return null;
  return sum / n < width / 2 ? "L" : "R";
}

function drawHands() {
  for (const hs of handStates) drawHand(hs);
}

function gloveHeight() {
  return Math.max(90, Math.min(320, height * 0.19));
}

// Extra reach the glove adds around the tracked wrist when hitting a muñeco.
function gloveHitPad() {
  return gloveHeight() * 0.22;
}

function drawHand(hs) {
  const now = millis();
  const hot = now < hs.punchUntil;
  const side = hs.side === "L" ? [255, 211, 110] : [168, 230, 207];
  const col = hot ? [255, 64, 84] : side;

  // Tracking stays visible all the time: a soft glow at the wrist.
  if (drawingContext) {
    drawingContext.shadowBlur = hot ? 30 : 14;
    drawingContext.shadowColor = `rgba(${col[0]}, ${col[1]}, ${col[2]}, 0.9)`;
  }
  noStroke();
  fill(col[0], col[1], col[2], hot ? 190 : 120);
  circle(hs.x, hs.y, hot ? 44 : 34);
  if (drawingContext) drawingContext.shadowBlur = 0;

  // The glove only shows up for a moment after a punch is detected.
  const gloveLeft = (hs.gloveUntil || 0) - now;
  if (gloveReady && gloveImg) {
    if (gloveLeft > 0) {
      const fade = Math.min(1, (gloveLeft / GLOVE_SHOW_MS) * 3);
      const grow = constrain((GLOVE_SHOW_MS - gloveLeft) / 90, 0.7, 1);
      drawGlove(hs, hot, fade, grow);
    }
  } else {
    // Fallback while the glove SVG loads.
    noStroke();
    fill(col[0], col[1], col[2], 235);
    circle(hs.x, hs.y, hot ? 22 : 15);
  }

  // Tracking ring on top, so the exact hit point stays readable.
  noFill();
  stroke(col[0], col[1], col[2], hot ? 235 : 160);
  strokeWeight(hot ? 3 : 2);
  circle(hs.x, hs.y, 18);
}

// Draws one glove half, cuff at the wrist, rotated with the forearm.
function drawGlove(hs, hot, alpha = 1, scale = 1) {
  const img = gloveImg;
  const src = img.canvas || img.elt; // p5.Image draws from its canvas
  if (!src) return;
  const crop = hs.limb === "left" ? GLOVE_CROP.left : GLOVE_CROP.right;

  const sx = crop.sx * img.width;
  const sy = crop.sy * img.height;
  const sw = crop.sw * img.width;
  const sh = crop.sh * img.height;
  if (sw <= 0 || sh <= 0) return;

  const h = gloveHeight() * (hot ? 1.07 : 1) * scale;
  const w = h * (sw / sh);
  const dy = -h * 0.92; // the cuff sits on the tracked wrist
  const prevAlpha = drawingContext.globalAlpha;

  push();
  translate(hs.x, hs.y);
  rotate(hs.angle || 0);
  drawingContext.globalAlpha = alpha;
  drawingContext.drawImage(src, sx, sy, sw, sh, -w / 2, dy, w, h);
  drawingContext.globalAlpha = prevAlpha;
  pop();
}

// --- Muñecos (art characters that pop out of and back into the tree holes) -
//
// Each one emerges from a free hole, rests briefly, then retreats back into
// it. Side holes draw the character rotated +90° / -90°.

function updateMunecos() {
  const now = millis();

  // Remove finished characters first so their holes free up immediately.
  for (let i = munecos.length - 1; i >= 0; i--) {
    const m = munecos[i];
    m.age = now - m.bornAt;
    if (m.age >= m.lifetime) munecos.splice(i, 1);
  }

  if (now >= nextSpawnAt) {
    spawnMuneco();
    nextSpawnAt = now + randomSpawnDelay();
  }
}

function randomSpawnDelay() {
  // Random interval that drifts over time; faster overall than before.
  const t = millis() / 1000;
  const drift = 0.5 + 0.5 * Math.sin(t / 5); // 0..1 oscillation (~31s)
  const lo = lerp(350, 800, drift);
  return random(lo, lo + 600);
}

// Background cast for the start screen: same emerge/idle/retreat cycle, but it
// never takes hits and only lives in bgMunecos.
function updateBgMunecos() {
  const now = millis();

  for (let i = bgMunecos.length - 1; i >= 0; i--) {
    const m = bgMunecos[i];
    m.age = now - m.bornAt;
    if (m.age >= m.lifetime) bgMunecos.splice(i, 1);
  }

  if (now >= bgNextSpawnAt) {
    spawnMuneco(bgMunecos);
    bgNextSpawnAt = now + random(650, 1500);
  }
}

function spawnMuneco(arr = munecos) {
  // Only use a hole that isn't already occupied (avoids overlapping designs).
  const used = new Set(arr.map((m) => m.hole));
  const free = MUNECO_HOLES.map((_, i) => i).filter((i) => !used.has(i));
  if (free.length === 0) return;

  const hole = free[Math.floor(random(free.length))];
  const spawn = MUNECO_HOLES[hole];
  const type = Math.floor(random(CHAR_FILES.length));
  const h = height * random(0.14, 0.18); // character height (~14-18% of screen)
  const w = h * charAspect(type);
  const popDuration = 380;
  // +1s of extra idle time so players have time to react before it retreats.
  const idleDuration = random(1400, 2200);

  // Map the hole (normalized to the SVG viewBox) straight onto the screen.
  const holeX = spawn.x * width;
  const holeY = spawn.y * height;

  // Side holes slide inward toward the screen center so they clear the trunk;
  // bottom holes grow straight out of the middle of their hole, no drift.
  const isBottom = spawn.y > 0.7;
  const dx = isBottom ? 0 : width / 2 - holeX;
  const dy = isBottom ? 0 : height / 2 - holeY;
  const len = Math.hypot(dx, dy) || 1;
  const popDist = isBottom ? 0 : h * 0.35;

  // Keep the whole character on screen. Side holes draw it rotated, so there
  // the on-screen box is h wide by w tall instead of w by h.
  const rotation = holeRotation(spawn.x);
  const sideways = Math.abs(rotation) > 0.1;
  const boxW = sideways ? h : w;
  const boxH = sideways ? w : h;
  const margin = 6;
  const halfW = boxW / 2 + margin;
  const halfH = boxH / 2 + margin;
  const targetX = constrain(holeX + (dx / len) * popDist, halfW, width - halfW);
  const targetY = constrain(holeY + (dy / len) * popDist, halfH, height - halfH);

  arr.push({
    type, // random character
    hole,
    h,
    rotation,
    startX: holeX,
    startY: holeY,
    targetX,
    targetY,
    bornAt: millis(),
    popDuration,
    idleDuration,
    lifetime: popDuration * 2 + idleDuration,
    age: 0,
  });
}

function drawMunecos() {
  for (const m of munecos) drawMuneco(m);
}

// Current on-screen pose of a muñeco (position + drawn size), shared by drawing,
// hit detection and the impact effect.
function munecoPose(m) {
  const age = m.age;
  const out = m.popDuration;
  const idleEnd = out + m.idleDuration;

  let scale, x, y;

  if (age < out) {
    // Coming out of the hole.
    const t = constrain(age / out, 0, 1);
    const e = easeOutBack(t);
    scale = e;
    x = lerp(m.startX, m.targetX, e);
    y = lerp(m.startY, m.targetY, e);
  } else if (age < idleEnd) {
    // Resting at full size.
    scale = 1;
    x = m.targetX;
    y = m.targetY;
  } else {
    // Going back into the hole (the frame on top hides it, no fade).
    const t = constrain((age - idleEnd) / out, 0, 1);
    const e = easeInBack(t);
    scale = 1 - e;
    x = lerp(m.targetX, m.startX, e);
    y = lerp(m.targetY, m.startY, e);
  }

  const aspect = charAspect(m.type);
  return { x, y, w: m.h * aspect * scale, h: m.h * scale, scale };
}

function drawMuneco(m) {
  const img = charImgs[m.type];
  const p = munecoPose(m);

  if (img && img.width) {
    push();
    translate(p.x, p.y);
    rotate(m.rotation);
    imageMode(CENTER);
    image(img, 0, 0, p.w, p.h);
    pop();
  } else {
    // Fallback while the character image is still loading.
    noStroke();
    fill(255, 200, 80);
    circle(p.x, p.y, m.h * 0.4 * p.scale);
  }
}

// --- Fist hits ------------------------------------------------------------

const FIST_HIT_COOLDOWN_MS = 260; // max one knockout per hand per punch
const lastFistHitAt = {};

// Any moving fist (left or right player) that lands on a muñeco removes it
// instantly and spawns the impact effect.
function resolveFistHits() {
  const now = millis();

  for (const hs of handStates) {
    if (now >= hs.punchUntil) continue; // only a moving fist breaks a character
    if (now - (lastFistHitAt[hs.key] || -Infinity) < FIST_HIT_COOLDOWN_MS) continue;

    let best = -1;
    let bestDist = Infinity;
    for (let i = 0; i < munecos.length; i++) {
      const m = munecos[i];
      const p = munecoPose(m);
      if (p.scale < 0.35) continue; // still too small to be hit
      if (!handSweepHits(hs, m, p)) continue;
      const d = Math.hypot(hs.x - p.x, hs.y - p.y);
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    }

    if (best >= 0) {
      const m = munecos[best];
      const p = munecoPose(m);
      spawnHitEffect(p.x, p.y, m.h);
      playHitSound(); // same instant as the burst
      munecos.splice(best, 1);
      lastFistHitAt[hs.key] = now;

      // The punch scores for the player on the hand's side of the divider.
      if (hs.side === "L") scores.L += 10;
      else if (hs.side === "R") scores.R += 10;
    }
  }
}

// Sweeps the wrist from its previous to its current position so a fast punch
// cannot tunnel through a character between two pose frames.
function handSweepHits(hs, m, p) {
  for (let i = 0; i <= 3; i++) {
    const t = i / 3;
    if (fistInsideMuneco(lerp(hs.prevX, hs.x, t), lerp(hs.prevY, hs.y, t), m, p)) return true;
  }
  return false;
}

// Tests the wrist point against the muñeco's rotated bounding box. The box is
// widened by the glove radius so a visible punch connects more easily.
function fistInsideMuneco(fx, fy, m, p) {
  const dx = fx - p.x;
  const dy = fy - p.y;
  const c = Math.cos(-m.rotation);
  const s = Math.sin(-m.rotation);
  const lx = dx * c - dy * s;
  const ly = dx * s + dy * c;
  const pad = gloveHitPad();
  return Math.abs(lx) <= p.w * 0.58 + pad && Math.abs(ly) <= p.h * 0.62 + pad;
}

// --- Impact effect (comic burst) ------------------------------------------

function spawnHitEffect(x, y, size) {
  effects.push({
    x,
    y,
    size: size * 1.45, // burst is a bit larger than the character
    rot: random(-0.3, 0.3), // varied angle per hit
    bornAt: millis(),
    duration: 460,
  });
}function updateEffects() {
  const now = millis();
  for (let i = effects.length - 1; i >= 0; i--) {
    if (now - effects[i].bornAt >= effects[i].duration) effects.splice(i, 1);
  }
}

function drawEffects() {
  if (!burstImg || !burstImg.width) return;

  const now = millis();
  const ar = burstImg.width / burstImg.height;

  for (const e of effects) {
    const t = constrain((now - e.bornAt) / e.duration, 0, 1);

    // Pop out fast, then fade the last half.
    const grow = easeOutCubic(t);
    const scale = lerp(0.35, 1.15, grow);
    const fade = 1 - constrain((t - 0.5) / 0.5, 0, 1);

    push();
    translate(e.x, e.y);
    rotate(e.rot + t * 0.25);
    tint(255, 255 * fade);
    imageMode(CENTER);
    image(burstImg, 0, 0, e.size * scale * ar, e.size * scale);
    pop();
  }

  noTint();
}

// --- Hit sound ------------------------------------------------------------
//
// A small pool of preloaded Audio elements so several punches can overlap and
// each hit can restart a free voice immediately.

const HIT_SOUND_URL = "golpe.mp3";
const HIT_VOICES = 4;
const hitVoices = [];
let hitVoiceIndex = 0;

function initHitSound() {
  for (let i = 0; i < HIT_VOICES; i++) {
    const a = new Audio(HIT_SOUND_URL);
    a.preload = "auto";
    a.volume = 0.85;
    hitVoices.push(a);
  }
}

function playHitSound() {
  if (!hitVoices.length) return;
  const a = hitVoices[hitVoiceIndex];
  hitVoiceIndex = (hitVoiceIndex + 1) % HIT_VOICES;
  try {
    a.currentTime = 0;
    const p = a.play();
    if (p && p.catch) p.catch(() => {}); // autoplay policy: ignore until a gesture
  } catch (err) {
    // Never let audio break the game loop.
  }
}

function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

function easeOutBack(t) {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
}

function easeInBack(t) {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return c3 * t * t * t - c1 * t * t;
}

// --- Helpers --------------------------------------------------------------

// Maps a raw x coordinate into screen space, honoring mirror mode.
function px(x) {
  return mirror ? width - x : x;
}

// ml5 renames keypoint `score` to `confidence`; read whichever is present.
function kpConfidence(kp) {
  return kp.confidence !== undefined ? kp.confidence : kp.score;
}

function keyPressed() {
  if (key === "m" || key === "M") mirror = !mirror;
  if (key === "r" || key === "R") initCamera();
}
