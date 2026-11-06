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

// Placeholder impact effects (expanding ring + sparks) shown when a fist hits.
let effects = [];

// Score per player (left / right of the divider), +10 for every character hit.
let scores = { L: 0, R: 0 };

// UI font, loaded from Google Fonts in index.html.
const SCORE_FONT = "Mochiy Pop One";

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
    document.fonts.load(`16px "${SCORE_FONT}"`).catch(() => {});
  }

  // Load the nature frame image asynchronously; draw it once it's ready.
  loadImage("ambiente.svg", onImageLoaded, onImageError);

  // Load the 3 character SVGs.
  CHAR_FILES.forEach((f, i) => {
    loadImage(f, (img) => { charImgs[i] = img; }, () => console.warn("[tracker] no cargó " + f));
  });

  // ml5 1.4 returns a Promise that resolves to the model, and only the model
  // has detectStart(). Older builds return the model directly, so handle both.
  loadPoseModel(ml5.bodyPose("MoveNet", { modelType: "MULTIPOSE_LIGHTNING" }), "bodyPose", onBodyPoseReady);
  loadPoseModel(ml5.handPose({ maxHands: 4 }), "handPose", onHandPoseReady);

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

// --- Score -----------------------------------------------------------------

function drawScores() {
  const size = Math.max(26, width * 0.045);
  const margin = Math.max(14, width * 0.035);

  push();
  textFont(SCORE_FONT);
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

function spawnMuneco() {
  // Only use a hole that isn't already occupied (avoids overlapping designs).
  const used = new Set(munecos.map((m) => m.hole));
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

  munecos.push({
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
    if (hs.gesture.closed < 3) continue; // only a closed fist counts
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

// --- Impact effect (placeholder) ------------------------------------------

function spawnHitEffect(x, y, size) {
  effects.push({ x, y, size, bornAt: millis(), duration: 420 });
}

function updateEffects() {
  const now = millis();
  for (let i = effects.length - 1; i >= 0; i--) {
    if (now - effects[i].bornAt >= effects[i].duration) effects.splice(i, 1);
  }
}

function drawEffects() {
  const now = millis();

  for (const e of effects) {
    const t = constrain((now - e.bornAt) / e.duration, 0, 1);
    const r = lerp(e.size * 0.18, e.size * 0.72, easeOutCubic(t));
    const a = 1 - t;

    // Expanding ring.
    noFill();
    stroke(255, 226, 138, 240 * a);
    strokeWeight(lerp(7, 1, t));
    circle(e.x, e.y, r * 2);

    // Sparks.
    stroke(255, 255, 255, 220 * a);
    strokeWeight(2);
    for (let k = 0; k < 8; k++) {
      const ang = (k / 8) * TWO_PI + t * 0.5;
      const r0 = r * 0.85;
      const r1 = r * (1.05 + 0.35 * (1 - t));
      line(e.x + Math.cos(ang) * r0, e.y + Math.sin(ang) * r0,
           e.x + Math.cos(ang) * r1, e.y + Math.sin(ang) * r1);
    }
  }

  noStroke();
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
}
