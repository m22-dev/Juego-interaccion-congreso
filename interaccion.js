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

// Placeholder "muñecos" that pop out of the nature frame's holes.
let munecos = [];
let nextSpawnAt = 0;
const MUNECO_COLORS = [
  [110, 180, 90],
  [220, 150, 70],
  [200, 100, 80],
  [90, 160, 170],
  [230, 180, 60],
];

// Nature frame (the ambience image) drawn fullscreen over the camera. Its
// transparent center shows the camera/players; muñecos spawn from its inner
// edge (the "holes").
let ambientImg;
let imgReady = false;
let imgW = 2560;
let imgH = 1540;
let imgScale = 1;
let imgOffX = 0;
let imgOffY = 0;

// Normalized center of the frame opening + spawn points along its inner edge
// (detected from the PNG's alpha channel).
const FRAME_CENTER = { x: 0.524, y: 0.49 };
const MUNECO_SPAWNS = [
  { x: 0.131, y: 0.49 }, { x: 0.131, y: 0.385 }, { x: 0.144, y: 0.302 }, { x: 0.188, y: 0.219 },
  { x: 0.237, y: 0.219 }, { x: 0.269, y: 0.208 }, { x: 0.356, y: 0.219 }, { x: 0.394, y: 0.198 },
  { x: 0.431, y: 0.177 }, { x: 0.487, y: 0.156 }, { x: 0.544, y: 0.156 }, { x: 0.575, y: 0.219 },
  { x: 0.631, y: 0.198 }, { x: 0.675, y: 0.198 }, { x: 0.713, y: 0.219 }, { x: 0.756, y: 0.24 },
  { x: 0.825, y: 0.198 }, { x: 0.856, y: 0.219 }, { x: 0.881, y: 0.281 }, { x: 0.9, y: 0.375 },
  { x: 0.9, y: 0.479 }, { x: 0.9, y: 0.583 }, { x: 0.9, y: 0.677 }, { x: 0.881, y: 0.771 },
  { x: 0.831, y: 0.792 }, { x: 0.769, y: 0.781 }, { x: 0.769, y: 0.813 }, { x: 0.675, y: 0.76 },
  { x: 0.619, y: 0.76 }, { x: 0.575, y: 0.813 }, { x: 0.525, y: 0.833 }, { x: 0.469, y: 0.833 },
  { x: 0.406, y: 0.833 }, { x: 0.4, y: 0.75 }, { x: 0.356, y: 0.708 }, { x: 0.294, y: 0.708 },
  { x: 0.2, y: 0.74 }, { x: 0.181, y: 0.729 }, { x: 0.144, y: 0.677 }, { x: 0.131, y: 0.594 },
];

function setup() {
  const cnv = createCanvas(windowWidth, windowHeight);
  cnv.parent("app");

  // Load the nature frame image asynchronously; draw it once it's ready.
  loadImage("ambiente.png", onImageLoaded, onImageError);

  bodyPose = ml5.bodyPose("MoveNet", { modelType: "MULTIPOSE_LIGHTNING" }, onBodyPoseReady);
  handPose = ml5.handPose({ maxHands: 4 }, onHandPoseReady);

  initCamera();

  nextSpawnAt = millis() + randomSpawnDelay();

  // Diagnostics: warn if a model never becomes ready (network/CORS issue).
  window.setTimeout(() => {
    if (!bodyReady) console.warn("[tracker] bodyPose no cargó en 15s (¿red/TFHub bloqueado?)");
    if (!handReady) console.warn("[tracker] handPose no cargó en 15s (¿red/CDN bloqueado?)");
  }, 15000);
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  updateCoverTransform();
  updateImageTransform();
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

// Cover transform for the nature frame image (fills the screen, crops overflow).
function updateImageTransform() {
  const ia = imgW / imgH;
  const ca = width / height;
  if (ca > ia) {
    imgScale = height / imgH;
    imgOffX = (width - imgW * imgScale) / 2;
    imgOffY = 0;
  } else {
    imgScale = width / imgW;
    imgOffX = 0;
    imgOffY = (height - imgH * imgScale) / 2;
  }
}

// Maps a normalized image coordinate (0..1) to screen space.
function mapImageToScreen(nx, ny) {
  return { x: nx * imgW * imgScale + imgOffX, y: ny * imgH * imgScale + imgOffY };
}

function onImageLoaded(img) {
  ambientImg = img; // p5 2.x delivers the real image via the callback
  imgW = img.width;
  imgH = img.height;
  imgReady = true;
  updateImageTransform();
  console.log("[tracker] ambiente cargado", imgW, "x", imgH);
}

function onImageError() {
  console.warn("[tracker] no se pudo cargar ambiente.png");
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

function onBodyPoseReady() {
  bodyReady = true;
  console.log("[tracker] bodyPose listo");
  tryStartBody();
}

function onHandPoseReady() {
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

  if (videoReady) {
    // Draw the camera feed fullscreen (mirrored if requested).
    push();
    if (mirror) {
      translate(width, 0);
      scale(-1, 1);
    }
    image(video, coverOffX, coverOffY, videoW * coverScale, videoH * coverScale);
    pop();
  }

  if (videoReady && bodyStarted) drawBodies();
  if (videoReady && handStarted) drawHands();

  // Nature frame drawn over the camera (its transparent center shows players).
  if (imgReady && ambientImg) {
    image(ambientImg, imgOffX, imgOffY, imgW * imgScale, imgH * imgScale);
  }

  updateMunecos();
  drawMunecos();

  if (videoReady) drawDivider();

  drawStatusDot();
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

function drawHands() {
  for (const hand of hands) drawHand(hand);
}

function drawHand(hand) {
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

  const col = gestureColor(handGesture(byName).label);

  // Subtle skeleton lines.
  stroke(col[0], col[1], col[2], 110);
  strokeWeight(1.2);
  for (const [a, b] of handConnections) {
    const pa = sm[a];
    const pb = sm[b];
    if (pa && pb) line(px(pa.x), pa.y, px(pb.x), pb.y);
  }

  // Small landmarks.
  noStroke();
  fill(col[0], col[1], col[2], 180);
  for (const kp of sm) circle(px(kp.x), kp.y, 5);
}

// Detects fist vs open hand by comparing fingertip-to-wrist distance against
// pip-to-wrist distance for each of the four fingers (not the thumb).
function handGesture(byName) {
  const wrist = byName["wrist"];
  if (!wrist) return { label: "?", closed: 0, open: 0 };

  const fingers = [
    ["index_finger_pip", "index_finger_tip"],
    ["middle_finger_pip", "middle_finger_tip"],
    ["ring_finger_pip", "ring_finger_tip"],
    ["pinky_pip", "pinky_tip"],
  ];

  let closed = 0;
  let open = 0;
  for (const [pipName, tipName] of fingers) {
    const pip = byName[pipName];
    const tip = byName[tipName];
    if (!pip || !tip) continue;
    const dTip = Math.hypot(tip.x - wrist.x, tip.y - wrist.y);
    const dPip = Math.hypot(pip.x - wrist.x, pip.y - wrist.y);
    if (dTip < dPip) closed++;
    else open++;
  }

  let label;
  if (closed >= 3) label = "PUÑO";
  else if (open >= 3) label = "MANO ABIERTA";
  else label = "SEMI";

  return { label, closed, open };
}

function gestureColor(label) {
  if (label === "PUÑO") return [255, 59, 78]; // red
  if (label === "MANO ABIERTA") return [46, 224, 106]; // green
  return [255, 176, 32]; // amber
}

// --- Muñecos (placeholder characters spawning from the frame holes) -------
//
// Simple colored blobs for now; they emerge from the frame's inner edge and
// pop inward. Later they'll be replaced by the real characters.

function updateMunecos() {
  const now = millis();
  if (now >= nextSpawnAt) {
    spawnMuneco();
    nextSpawnAt = now + randomSpawnDelay();
  }
  for (let i = munecos.length - 1; i >= 0; i--) {
    const m = munecos[i];
    m.age = now - m.bornAt;
    if (m.age >= m.lifetime) munecos.splice(i, 1);
  }
}

function randomSpawnDelay() {
  // Random interval that drifts over time; faster overall than before.
  const t = millis() / 1000;
  const drift = 0.5 + 0.5 * Math.sin(t / 5); // 0..1 oscillation (~31s)
  const lo = lerp(350, 800, drift);
  return random(lo, lo + 600);
}

function spawnMuneco() {
  const r = random(32, 52); // 2x bigger placeholder
  const spawn = MUNECO_SPAWNS[Math.floor(random(MUNECO_SPAWNS.length))];
  const start = mapImageToScreen(spawn.x, spawn.y);
  const center = mapImageToScreen(FRAME_CENTER.x, FRAME_CENTER.y);

  // Pop inward from the hole toward the center of the frame opening.
  const dx = center.x - start.x;
  const dy = center.y - start.y;
  const len = Math.hypot(dx, dy) || 1;
  const popDist = r + 46;

  munecos.push({
    startX: start.x,
    startY: start.y,
    targetX: start.x + (dx / len) * popDist,
    targetY: start.y + (dy / len) * popDist,
    r,
    col: MUNECO_COLORS[Math.floor(random(MUNECO_COLORS.length))],
    bornAt: millis(),
    lifetime: random(1000, 3500), // minimum 1 second
    popDuration: 280,
    age: 0,
  });
}

function drawMunecos() {
  for (const m of munecos) drawMuneco(m);
}

function drawMuneco(m) {
  const pop = easeOutBack(constrain(m.age / m.popDuration, 0, 1));
  const x = lerp(m.startX, m.targetX, pop);
  const y = lerp(m.startY, m.targetY, pop);

  let alpha = 255;
  if (m.age > m.lifetime - 250) {
    alpha = map(m.age, m.lifetime - 250, m.lifetime, 255, 0);
  }

  const [cr, cg, cb] = m.col;
  const r = m.r;
  noStroke();

  // Body.
  fill(cr, cg, cb, alpha);
  circle(x, y, r * 2);

  // Face highlight.
  fill(255, 255, 255, alpha * 0.35);
  circle(x - r * 0.22, y - r * 0.3, r * 0.55);

  // Eyes.
  fill(45, 32, 24, alpha);
  circle(x - r * 0.28, y - r * 0.12, r * 0.22);
  circle(x + r * 0.28, y - r * 0.12, r * 0.22);
}

function easeOutBack(t) {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
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
}
