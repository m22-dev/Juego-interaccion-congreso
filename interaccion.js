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

// Muñecos: historical art characters that pop out of the tree holes.
let munecos = [];
let nextSpawnAt = 0;

// The 3 character portraits (Kandinsky, Shakespeare, Van Gogh), loaded async.
const CHAR_FILES = ["kandinsky.svg", "shakespeare.svg", "vangogh.svg"];
const charImgs = [];

let ambientImg;
let imgReady = false;

// The tree holes (normalized to the SVG viewBox 2573.89 x 1562.94) that the
// muñecos pop out of: 2 left, 1 right-upper, and 2 at the bottom (new).
const MUNECO_HOLES = [
  { x: 0.022, y: 0.371 }, // left-upper
  { x: 0.084, y: 0.552 }, // left-lower
  { x: 0.963, y: 0.356 }, // right-upper
  { x: 0.294, y: 0.784 }, // bottom-left (new)
  { x: 0.666, y: 0.83 },  // bottom-right (new)
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

  // Load the nature frame image asynchronously; draw it once it's ready.
  loadImage("ambiente.svg", onImageLoaded, onImageError);

  // Load the 3 character SVGs.
  CHAR_FILES.forEach((f, i) => {
    loadImage(f, (img) => { charImgs[i] = img; }, () => console.warn("[tracker] no cargó " + f));
  });

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

  updateMunecos();
  drawMunecos(); // drawn under the frame so they hide inside the trunk

  // Nature frame drawn on top (covers the muñecos while they're in the trunk).
  if (imgReady && ambientImg) {
    image(ambientImg, 0, 0, width, height);
  }

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

// --- Muñecos (art characters that pop out of and back into the tree holes) -
//
// Each one emerges from a free hole, rests briefly, then retreats back into
// it. Left-side characters are rotated +45°, right-side -45°.

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

function spawnMuneco() {
  // Only use a hole that isn't already occupied (avoids overlapping designs).
  const used = new Set(munecos.map((m) => m.hole));
  const free = MUNECO_HOLES.map((_, i) => i).filter((i) => !used.has(i));
  if (free.length === 0) return;

  const hole = free[Math.floor(random(free.length))];
  const spawn = MUNECO_HOLES[hole];
  const h = height * random(0.14, 0.18); // character height (~14-18% of screen)
  const popDuration = 380;
  const idleDuration = random(400, 1200);

  // Map the hole (normalized to the SVG viewBox) straight onto the screen.
  const startX = spawn.x * width;
  const startY = spawn.y * height;

  // Slide inward from the hole toward the center of the screen.
  const dx = width / 2 - startX;
  const dy = height / 2 - startY;
  const len = Math.hypot(dx, dy) || 1;
  const popDist = h * 0.35;

  munecos.push({
    type: Math.floor(random(CHAR_FILES.length)), // random character
    hole,
    h,
    rotation: holeRotation(spawn.x),
    startX,
    startY,
    targetX: startX + (dx / len) * popDist,
    targetY: startY + (dy / len) * popDist,
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

function drawMuneco(m) {
  const img = charImgs[m.type];
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

  if (img && img.width) {
    const w = m.h * (img.width / img.height);
    push();
    translate(x, y);
    rotate(m.rotation);
    imageMode(CENTER);
    image(img, 0, 0, w * scale, m.h * scale);
    pop();
  } else {
    // Fallback while the character image is still loading.
    noStroke();
    fill(255, 200, 80);
    circle(x, y, m.h * 0.4 * scale);
  }
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
}
