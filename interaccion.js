// Arm & Fist Tracker — 2-player
// Webcam + ml5.js (bodyPose "MoveNet" MultiPose + handPose) + p5.js
//
// Fullscreen camera feed with a subtle skeleton for the first two people on
// screen, a subtle vertical divider at the center, and gesture-tinted hands.
// No on-screen text.

const SCORE_THRESHOLD = 0.3; // minimum confidence for a body keypoint

const CAM_W = 1280;
const CAM_H = 720;

let video;
let bodyPose;
let handPose;

let poses = [];
let hands = [];
let handStates = []; // smoothed per-hand data computed once per frame

let bodyConnections = [];
let handConnections = [];

let videoReady = false;
let bodyReady = false;
let handReady = false;
let bodyStarted = false;
let handStarted = false;

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
const handSmoothers = {}; // one smoother per hand (keyed by side + handedness)

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

// "start" shows the menu; "playing" runs the game.
let gameState = "start";

// Time the four fists must be held up before the match starts.
const START_HOLD_MS = 700;
let startHoldMs = 0;

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
let trackIndex = 0;
let bgmEl = null;

// Pointing selector: hold the index fingertip over a pill to pick that song.
const POINT_HOLD_MS = 700;
let pointTarget = -1;
let pointHoldMs = 0;
let pointCooldownUntil = 0;

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

  // Preload the punch sound (a small pool so hits can overlap).
  initHitSound();

  // Grab the shared background-music element (its src is set by index.html).
  initMusic();

  // ml5 1.4 returns a Promise that resolves to the model, and only the model
  // has detectStart(). Older builds return the model directly, so handle both.
  loadPoseModel(ml5.bodyPose("MoveNet", { modelType: "MULTIPOSE_LIGHTNING" }), "bodyPose", onBodyPoseReady);
  loadPoseModel(ml5.handPose({ maxHands: 4 }), "handPose", onHandPoseReady);

  initCamera();

  nextSpawnAt = millis() + randomSpawnDelay();
  bgNextSpawnAt = millis() + 350;

  // Diagnostics: warn if a model never becomes ready (network/CORS issue).
  window.setTimeout(() => {
    if (!bodyReady) console.warn("[tracker] bodyPose no cargó en 15s (¿red/TFHub bloqueado?)");
    if (!handReady) console.warn("[tracker] handPose no cargó en 15s (¿red/CDN bloqueado?)");
  }, 15000);
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  updateCoverTransform();
}

// --- Camera ---------------------------------------------------------------

function initCamera() {
  if (bodyStarted) { bodyPose.detectStop(); bodyStarted = false; }
  if (handStarted) { handPose.detectStop(); handStarted = false; }
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

// --- Readiness (body and hand start independently) ------------------------

function onVideoReady() {
  videoW = video.elt.videoWidth || CAM_W;
  videoH = video.elt.videoHeight || CAM_H;
  updateCoverTransform();
  videoReady = true;
  tryStartBody();
  tryStartHand();
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

function onHandPoseReady(model) {
  if (model) handPose = model;
  handReady = true;
  console.log("[tracker] handPose listo");
  tryStartHand();
}

function tryStartBody() {
  if (bodyStarted || !videoReady || !bodyReady) return;
  bodyStarted = true;
  bodyPose.detectStart(video, gotPoses);
  bodyConnections = bodyPose.getConnections() || [];
  console.log("[tracker] bodyPose detectStart iniciado");
}

function tryStartHand() {
  if (handStarted || !videoReady || !handReady) return;
  handStarted = true;
  handPose.detectStart(video, gotHands);
  handConnections = handPose.getConnections() || [];
  console.log("[tracker] handPose detectStart iniciado");
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

function gotHands(results) {
  hands = results || [];
  for (const hand of hands) {
    for (const kp of hand.keypoints) {
      kp.x = kp.x * coverScale + coverOffX;
      kp.y = kp.y * coverScale + coverOffY;
    }
  }
}

// --- Main loop ------------------------------------------------------------

function draw() {
  background(0);

  if (gameState === "start") {
    drawStartScreen();
    return;
  }

  drawCameraFeed();
  if (videoReady && bodyStarted) drawBodies();

  // Hands are smoothed once per frame, then fist hits are resolved against the
  // muñecos before drawing so a punched character vanishes immediately.
  if (videoReady && handStarted) updateHands();

  updateMunecos();
  if (videoReady && handStarted) resolveFistHits();
  updateEffects();

  if (videoReady && handStarted) drawHands();

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
  if (videoReady && handStarted) {
    updateHands();
    drawHands();
  }

  const gate = updateStartGate();

  drawStartTitle();

  // Two short lines so it also fits on narrow screens.
  const lineSize = Math.max(12, width * 0.024);
  const line1Y = height * 0.375;
  const line2Y = line1Y + lineSize * 1.5;
  drawMenuText("CADA JUGADOR ACERCA", width / 2, line1Y, lineSize, [255, 246, 226]);
  drawMenuText("SUS 2 PUÑOS A LA CÁMARA", width / 2, line2Y, lineSize, [255, 246, 226]);

  // Either the fist indicators (when tracking works) or a short status line.
  if (videoReady && handStarted) {
    drawFistDots(gate);
  } else {
    drawCameraStatus(height * 0.465);
  }

  // Song picker: pills laid out first so the pointing test can use them.
  const layout = trackPickerLayout();
  const point = updatePointing(layout);
  drawTrackPicker(layout, point);
  drawPointCursor(point);
}

function drawStartTitle() {
  push();
  textAlign(CENTER, CENTER);
  textFont(UI_FONT);
  textSize(Math.max(42, width * 0.115));

  // Soft shadow.
  noStroke();
  fill(16, 7, 14, 130);
  text("EMPEZAR", width / 2 + 3, height * 0.255 + 5);

  // Same artwork as Empezar_Texto-03.svg: Mochiy Pop One in #8a336a, with a
  // cream edge so it stays readable over the live camera.
  stroke(255, 243, 210);
  strokeWeight(Math.max(5, width * 0.0095));
  fill(138, 51, 106);
  text("EMPEZAR", width / 2, height * 0.255);
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

// Reads the four hands and decides when both players are showing two fists.
function updateStartGate() {
  const fists = handStates.filter((hs) => hs.gesture.closed >= 3 && !hs.gesture.pointing);
  const left = fists.filter((hs) => hs.side === "L").length;
  const right = fists.filter((hs) => hs.side === "R").length;
  const ready = left >= 2 && right >= 2;
  const dt = Math.min(deltaTime, 100);

  if (ready) startHoldMs += dt;
  else startHoldMs = Math.max(0, startHoldMs - dt * 2);

  if (startHoldMs >= START_HOLD_MS) startGame();

  return {
    left,
    right,
    ready,
    progress: constrain(startHoldMs / START_HOLD_MS, 0, 1),
  };
}

function drawFistDots(gate) {
  const cx = width / 2;
  const cy = height * 0.465;
  const r = Math.max(6, width * 0.009);
  const gap = r * 3.4;
  const xs = [cx - gap * 1.65, cx - gap * 0.62, cx + gap * 0.62, cx + gap * 1.65];

  const lit = [gate.left >= 1, gate.left >= 2, gate.right >= 1, gate.right >= 2];
  const gold = [255, 211, 110];
  const mint = [168, 230, 207];
  const cols = [gold, gold, mint, mint];

  strokeWeight(2);
  for (let i = 0; i < 4; i++) {
    if (lit[i]) {
      noStroke();
      fill(cols[i][0], cols[i][1], cols[i][2]);
      circle(xs[i], cy, r * 2);
    } else {
      noFill();
      stroke(255, 245, 220, 170);
      circle(xs[i], cy, r * 2);
    }
  }

  // Fill bar while the four fists are held.
  if (gate.progress > 0) {
    const bw = gap * 4.4;
    const bh = Math.max(4, height * 0.006);
    const by = cy + r * 2.4;
    noStroke();
    fill(255, 245, 220, 55);
    rect(cx - bw / 2, by, bw, bh, bh / 2);
    fill(255, 226, 138, 235);
    rect(cx - bw / 2, by, bw * gate.progress, bh, bh / 2);
  }
}

function drawCameraStatus(y) {
  let msg = null;
  if (!videoReady) msg = "ESPERANDO CÁMARA…";
  else if (!handStarted) msg = "CARGANDO DETECCIÓN…";
  if (!msg) return;
  drawMenuText(msg, width / 2, y, Math.max(10, width * 0.017), [255, 236, 200]);
}

function drawTrackPicker(layout, point) {
  const labelSize = Math.max(10, width * 0.016);
  drawMenuText(
    "ELEGÍ LA MÚSICA APUNTANDO CON EL DEDO",
    width / 2,
    layout[0].y - labelSize * 1.9,
    labelSize,
    [255, 240, 205]
  );

  for (let i = 0; i < layout.length; i++) {
    const r = layout[i];
    const selected = i === trackIndex;
    const hovered = point && point.target === i;

    push();
    strokeWeight(2);
    if (selected) {
      fill(255, 211, 110, 240);
      stroke(43, 26, 16, 230);
    } else {
      fill(20, 12, 20, 155);
      stroke(255, 245, 220, 130);
    }
    rect(r.x, r.y, r.w, r.h, r.h / 2);

    // Pointing highlight + dwell progress inside the pill.
    if (hovered) {
      noFill();
      stroke(86, 204, 255, 230);
      strokeWeight(3);
      rect(r.x - 3, r.y - 3, r.w + 6, r.h + 6, r.h / 2 + 3);

      noStroke();
      if (selected) fill(43, 26, 16, 200);
      else fill(255, 211, 110, 220);
      rect(r.x + 13, r.y + r.h - 11, (r.w - 26) * point.progress, 4, 2);
    }

    noStroke();
    if (selected) fill(43, 26, 16);
    else fill(255, 245, 220);
    textAlign(CENTER, CENTER);
    textFont(UI_FONT);
    textSize(Math.max(9, Math.min(r.h * 0.3, r.w * 0.082)));
    text(i + 1 + "  " + TRACKS[i].name.toUpperCase(), r.x + r.w / 2, r.y + r.h / 2 + 1);
    pop();
  }
}

// Reads every pointing hand and accumulates dwell time over the pill under the
// index fingertip. Holding there for POINT_HOLD_MS selects that song.
function updatePointing(layout) {
  const now = millis();
  const dt = Math.min(deltaTime, 100);

  // Short lockout after a pick so it does not re-trigger immediately.
  if (now < pointCooldownUntil) {
    return { target: -1, progress: 0, x: 0, y: 0, hand: false };
  }

  let fingertip = null;
  let hit = -1;

  for (const hs of handStates) {
    if (!hs.gesture.pointing) continue;
    const tip = hs.byName["index_finger_tip"];
    if (!tip) continue;

    const x = px(tip.x);
    const y = tip.y;
    if (!fingertip) fingertip = { x, y };

    for (let i = 0; i < layout.length; i++) {
      const r = layout[i];
      if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) {
        if (hit < 0) {
          hit = i;
          fingertip = { x, y };
        }
        break;
      }
    }
  }

  if (hit >= 0) {
    if (hit === pointTarget) pointHoldMs += dt;
    else {
      pointTarget = hit;
      pointHoldMs = 0;
    }
  } else {
    pointTarget = -1;
    pointHoldMs = Math.max(0, pointHoldMs - dt * 2);
  }

  if (pointTarget >= 0 && pointHoldMs >= POINT_HOLD_MS) {
    selectTrack(pointTarget);
    pointCooldownUntil = now + 700;
    pointHoldMs = 0;
    pointTarget = -1;
    return { target: -1, progress: 0, x: fingertip ? fingertip.x : 0, y: fingertip ? fingertip.y : 0, hand: !!fingertip };
  }

  return {
    target: pointTarget,
    progress: pointTarget >= 0 ? constrain(pointHoldMs / POINT_HOLD_MS, 0, 1) : 0,
    x: fingertip ? fingertip.x : 0,
    y: fingertip ? fingertip.y : 0,
    hand: !!fingertip,
  };
}

function drawPointCursor(point) {
  // Over a pill the highlight + progress bar are the feedback; the cursor is
  // only shown while pointing at empty space.
  if (!point.hand || point.target >= 0) return;
  push();
  noFill();
  stroke(86, 204, 255, 210);
  strokeWeight(3);
  circle(point.x, point.y, 26);
  noStroke();
  fill(86, 204, 255, 240);
  circle(point.x, point.y, 8);
  pop();
}

// Vertical list of the three tracks, centered under the fist indicators.
function trackPickerLayout() {
  const ph = Math.max(34, Math.min(height * 0.052, 58));
  const gap = ph * 0.3;
  const pw = Math.min(width * 0.56, 460);
  const total = ph * 3 + gap * 2;
  // Never overflow the bottom on short screens.
  const y0 = Math.min(height * 0.58, height - total - 24);
  return TRACKS.map((_, i) => ({ x: (width - pw) / 2, y: y0 + i * (ph + gap), w: pw, h: ph }));
}

function startGame() {
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
  playMusic();
}

// --- Music ----------------------------------------------------------------

function initMusic() {
  bgmEl = document.getElementById("bgm");
}

function selectTrack(i) {
  if (i < 0 || i >= TRACKS.length) return;
  trackIndex = i;
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
  for (const kp of pose.keypoints) {
    // Per-person smoothing key so the two players don't share state.
    const p = bodySmooth.next(`${index}:${kp.name}`, kp.x, kp.y);
    sm.push({ ...kp, x: p.x, y: p.y });
  }

  // Subtle full-body skeleton.
  stroke(255, 255, 255, 110);
  strokeWeight(2);
  for (const [a, b] of bodyConnections) {
    const pa = sm[a];
    const pb = sm[b];
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

// --- Hands (fists) --------------------------------------------------------

function updateHands() {
  handStates = hands.map((hand) => handState(hand));
}

// Smooths one hand and classifies its gesture. Kept separate from drawing so the
// exact same (smoothed) positions are used for hit detection.
function handState(hand) {
  // Identify the player by wrist position relative to the center line
  // (players stay on their own side, so side is a stable ownership cue).
  const wristRaw = hand.keypoints.find((k) => k.name === "wrist");
  const side = wristRaw ? (px(wristRaw.x) < width / 2 ? "L" : "R") : "?";
  const key = `${side}:${handKey(hand)}`;
  if (!handSmoothers[key]) handSmoothers[key] = new Smooth(0.5);

  const sm = [];
  const byName = {};
  for (const kp of hand.keypoints) {
    const p = handSmoothers[key].next(kp.name, kp.x, kp.y, 100);
    const s = { ...kp, x: p.x, y: p.y };
    sm.push(s);
    byName[s.name] = s;
  }

  return { key, side, sm, byName, gesture: handGesture(byName) };
}

function drawHands() {
  for (const hs of handStates) drawHand(hs);
}

function drawHand(hs) {
  const col = gestureColor(hs.gesture.label);

  // Subtle skeleton lines.
  stroke(col[0], col[1], col[2], 110);
  strokeWeight(1.2);
  for (const [a, b] of handConnections) {
    const pa = hs.sm[a];
    const pb = hs.sm[b];
    if (pa && pb) line(px(pa.x), pa.y, px(pb.x), pb.y);
  }

  // Small landmarks.
  noStroke();
  fill(col[0], col[1], col[2], 180);
  for (const kp of hs.sm) circle(px(kp.x), kp.y, 5);
}

// Center of the fist: wrist plus the four knuckles. Used as the punch point.
function fistCenter(hs) {
  const parts = ["wrist", "index_finger_mcp", "middle_finger_mcp", "ring_finger_mcp", "pinky_mcp"];
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (const name of parts) {
    const kp = hs.byName[name];
    if (!kp) continue;
    sx += px(kp.x);
    sy += kp.y;
    n++;
  }
  return n ? { x: sx / n, y: sy / n } : null;
}

// Detects fist / open hand / pointing by comparing fingertip-to-wrist distance
// against pip-to-wrist distance for each of the four fingers (not the thumb).
function handGesture(byName) {
  const wrist = byName["wrist"];
  if (!wrist) return { label: "?", closed: 0, open: 0, pointing: false };

  const fingers = [
    ["index_finger_pip", "index_finger_tip", true],
    ["middle_finger_pip", "middle_finger_tip", false],
    ["ring_finger_pip", "ring_finger_tip", false],
    ["pinky_pip", "pinky_tip", false],
  ];

  let closed = 0;
  let open = 0;
  let indexOpen = false;
  for (const [pipName, tipName, isIndex] of fingers) {
    const pip = byName[pipName];
    const tip = byName[tipName];
    if (!pip || !tip) continue;
    const dTip = Math.hypot(tip.x - wrist.x, tip.y - wrist.y);
    const dPip = Math.hypot(pip.x - wrist.x, pip.y - wrist.y);
    if (dTip < dPip) {
      closed++;
    } else {
      open++;
      if (isIndex) indexOpen = true;
    }
  }

  // Index extended with the other three curled.
  const pointing = indexOpen && closed === 3;

  let label;
  if (pointing) label = "SEÑALANDO";
  else if (closed >= 3) label = "PUÑO";
  else if (open >= 3) label = "MANO ABIERTA";
  else label = "SEMI";

  return { label, closed, open, pointing };
}

function gestureColor(label) {
  if (label === "SEÑALANDO") return [86, 204, 255]; // cyan
  if (label === "PUÑO") return [255, 59, 78]; // red
  if (label === "MANO ABIERTA") return [46, 224, 106]; // green
  return [255, 176, 32]; // amber
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

// Any fist (left or right player) that lands on a muñeco removes it instantly and
// spawns a placeholder impact effect.
function resolveFistHits() {
  const now = millis();

  for (const hs of handStates) {
    if (hs.gesture.closed < 3 || hs.gesture.pointing) continue; // a pointing hand is not a punch
    const fist = fistCenter(hs);
    if (!fist) continue;
    if (now - (lastFistHitAt[hs.key] || -Infinity) < FIST_HIT_COOLDOWN_MS) continue;

    let best = -1;
    let bestDist = Infinity;
    for (let i = 0; i < munecos.length; i++) {
      const m = munecos[i];
      const p = munecoPose(m);
      if (p.scale < 0.35) continue; // still too small to be hit
      if (!fistInsideMuneco(fist.x, fist.y, m, p)) continue;
      const d = Math.hypot(fist.x - p.x, fist.y - p.y);
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

// Tests the fist point against the muñeco's rotated bounding box.
function fistInsideMuneco(fx, fy, m, p) {
  const dx = fx - p.x;
  const dy = fy - p.y;
  const c = Math.cos(-m.rotation);
  const s = Math.sin(-m.rotation);
  const lx = dx * c - dy * s;
  const ly = dx * s + dy * c;
  return Math.abs(lx) <= p.w * 0.5 && Math.abs(ly) <= p.h * 0.55;
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

function handKey(hand) {
  const h = hand.handedness;
  const s = typeof h === "string" ? h : h && h.label ? h.label : "";
  return s || "h";
}

function keyPressed() {
  if (key === "m" || key === "M") mirror = !mirror;
  if (key === "r" || key === "R") initCamera();

  // Song shortcuts on the start screen (pointing is the on-screen way).
  if (gameState === "start" && (key === "1" || key === "2" || key === "3")) {
    selectTrack(Number(key) - 1);
  }
}
