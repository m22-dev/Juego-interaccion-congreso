// Arm & Fist Tracker
// Webcam + ml5.js (bodyPose "MoveNet" + handPose) + p5.js
//
// Fullscreen camera feed with a skeleton overlay. Tracks the arms
// (shoulder -> elbow -> wrist) and tints each hand by gesture
// (fist = red, open = green, partial = amber). No on-screen text.

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
let started = false;

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

const bodySmooth = new Smooth(0.5);
const handSmoothers = {}; // one smoother per hand (keyed by handedness)

function setup() {
  const cnv = createCanvas(windowWidth, windowHeight);
  cnv.parent("app");

  bodyPose = ml5.bodyPose("MoveNet", onBodyPoseReady);
  handPose = ml5.handPose({ maxHands: 2 }, onHandPoseReady);

  initCamera();
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  updateCoverTransform();
}

// --- Camera ---------------------------------------------------------------

function initCamera() {
  if (started) {
    bodyPose.detectStop();
    handPose.detectStop();
    started = false;
  }
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

// --- Readiness ------------------------------------------------------------

function onVideoReady() {
  videoW = video.elt.videoWidth || CAM_W;
  videoH = video.elt.videoHeight || CAM_H;
  updateCoverTransform();
  videoReady = true;
  tryStart();
}

function onBodyPoseReady() {
  bodyReady = true;
  tryStart();
}

function onHandPoseReady() {
  handReady = true;
  tryStart();
}

function tryStart() {
  if (started || !videoReady || !bodyReady || !handReady) return;
  started = true;
  bodyPose.detectStart(video, gotPoses);
  handPose.detectStart(video, gotHands);
  bodyConnections = bodyPose.getConnections() || [];
  handConnections = handPose.getConnections() || [];
}

function gotPoses(results) {
  poses = results || [];
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

  if (started && videoReady) {
    drawBodies();
    drawHands();
  }
}

// --- Body (arms) ----------------------------------------------------------

function drawBodies() {
  for (const pose of poses) drawPose(pose);
}

function drawPose(pose) {
  const sm = [];
  const byName = {};
  for (const kp of pose.keypoints) {
    const p = bodySmooth.next(kp.name, kp.x, kp.y);
    const s = { ...kp, x: p.x, y: p.y };
    sm.push(s);
    byName[s.name] = s;
  }

  // Full skeleton (subtle).
  stroke(255, 255, 255, 55);
  strokeWeight(2);
  for (const [a, b] of bodyConnections) {
    const pa = sm[a];
    const pb = sm[b];
    if (pa && pb && pa.score > SCORE_THRESHOLD && pb.score > SCORE_THRESHOLD) {
      line(px(pa.x), pa.y, px(pb.x), pb.y);
    }
  }

  // Arms highlighted (left = cyan, right = pink).
  drawBone(byName, "left_shoulder", "left_elbow", "#35e0ff");
  drawBone(byName, "left_elbow", "left_wrist", "#35e0ff");
  drawBone(byName, "right_shoulder", "right_elbow", "#ff5b8a");
  drawBone(byName, "right_elbow", "right_wrist", "#ff5b8a");

  drawJoint(byName, "left_shoulder", "#35e0ff");
  drawJoint(byName, "left_elbow", "#35e0ff");
  drawJoint(byName, "left_wrist", "#35e0ff");
  drawJoint(byName, "right_shoulder", "#ff5b8a");
  drawJoint(byName, "right_elbow", "#ff5b8a");
  drawJoint(byName, "right_wrist", "#ff5b8a");
}

function drawBone(byName, a, b, col) {
  const pa = byName[a];
  const pb = byName[b];
  if (!pa || !pb || pa.score < SCORE_THRESHOLD || pb.score < SCORE_THRESHOLD) return;
  stroke(col);
  strokeWeight(8);
  line(px(pa.x), pa.y, px(pb.x), pb.y);
  stroke(255);
  strokeWeight(3);
  line(px(pa.x), pa.y, px(pb.x), pb.y);
}

function drawJoint(byName, name, col) {
  const p = byName[name];
  if (!p || p.score < SCORE_THRESHOLD) return;
  noStroke();
  fill(col);
  circle(px(p.x), p.y, 15);
  fill(255);
  circle(px(p.x), p.y, 6);
}

// --- Hands (fists) --------------------------------------------------------

function drawHands() {
  for (const hand of hands) drawHand(hand);
}

function drawHand(hand) {
  const key = handKey(hand);
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

  // Skeleton.
  stroke(col[0], col[1], col[2], 210);
  strokeWeight(2);
  for (const [a, b] of handConnections) {
    const pa = sm[a];
    const pb = sm[b];
    if (pa && pb) line(px(pa.x), pa.y, px(pb.x), pb.y);
  }

  // Landmarks.
  noStroke();
  fill(col[0], col[1], col[2]);
  for (const kp of sm) circle(px(kp.x), kp.y, 6);
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

// --- Helpers --------------------------------------------------------------

// Maps a raw x coordinate into screen space, honoring mirror mode.
function px(x) {
  return mirror ? width - x : x;
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
