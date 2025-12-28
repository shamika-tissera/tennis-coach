let currentSessionId = null;
let sessionData = null;
let chatHistory = [];

const DEFAULT_COCO17_BONES = [
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [0, 5],
  [0, 6],
];

const ui = {
  sessionSelect: document.getElementById("sessionSelect"),
  simValue: document.getElementById("simValue"),
  expertValue: document.getElementById("expertValue"),
  feedbackList: document.getElementById("feedbackList"),

  genMeshBtn: document.getElementById("genMeshBtn"),
  genMeshExpert: document.getElementById("genMeshExpert"),
  genMeshClip: document.getElementById("genMeshClip"),
  meshStatus: document.getElementById("meshStatus"),

  frameSlider: document.getElementById("frameSlider"),
  timeLabel: document.getElementById("timeLabel"),
  frameLabel: document.getElementById("frameLabel"),
  playBtn: document.getElementById("playBtn"),
  pauseBtn: document.getElementById("pauseBtn"),
  speedSelect: document.getElementById("speedSelect"),
  viewMode: document.getElementById("viewMode"),
  ghostToggle: document.getElementById("ghostToggle"),
  partsToggle: document.getElementById("partsToggle"),
  diffToggle: document.getElementById("diffToggle"),
  viewerNotice: document.getElementById("viewerNotice"),

  chatLog: document.getElementById("chatLog"),
  chatText: document.getElementById("chatText"),
  sendBtn: document.getElementById("sendBtn"),
  chatError: document.getElementById("chatError"),

  newSessionBtn: document.getElementById("newSessionBtn"),
  modal: document.getElementById("modal"),
  videoPath: document.getElementById("videoPath"),
  tImpact: document.getElementById("tImpact"),
  strokeType: document.getElementById("strokeType"),
  multiShot: document.getElementById("multiShot"),
  maxShots: document.getElementById("maxShots"),
  minSep: document.getElementById("minSep"),
  viewFilter: document.getElementById("viewFilter"),
  resampleLen: document.getElementById("resampleLen"),
  createBtn: document.getElementById("createBtn"),
  cancelBtn: document.getElementById("cancelBtn"),
  createStatus: document.getElementById("createStatus"),

  leftResizer: document.getElementById("leftResizer"),
  rightResizer: document.getElementById("rightResizer"),
};

function el(tag, className, text) {
  const n = document.createElement(tag);
  if (className) n.className = className;
  if (text !== undefined) n.textContent = text;
  return n;
}

function setError(text) {
  ui.chatError.textContent = text || "";
}

function setMeshStatus(text) {
  if (!ui.meshStatus) return;
  ui.meshStatus.textContent = text || "";
}

function escapeHTML(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function sanitizeHref(href) {
  const s = String(href || "").trim();
  if (!s) return null;
  if (s.startsWith("#") || s.startsWith("/")) return s;
  if (/^https?:\/\//i.test(s)) return s;
  if (/^mailto:/i.test(s)) return s;
  return null;
}

function applyInlineMarkdown(escapedText) {
  const codeSpans = [];
  let out = String(escapedText || "");
  out = out.replace(/`([^`]+)`/g, (_, code) => {
    const idx = codeSpans.length;
    codeSpans.push(`<code>${code}</code>`);
    return `@@CODE${idx}@@`;
  });

  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/\*([^*]+)\*/g, "<em>$1</em>");

  out = out.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, url) => {
    const href = sanitizeHref(url);
    if (!href) return `${label} (${url})`;
    return `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`;
  });

  out = out.replace(/@@CODE(\d+)@@/g, (_, n) => codeSpans[Number(n)] || "");
  return out;
}

function isTableSeparatorLine(line) {
  const s = String(line || "").trim();
  if (!s) return false;
  // Matches: |---|---| or ---|--- (optionally with colons for alignment)
  return /^(\|?\s*:?-{3,}:?\s*)+(\|\s*:?-{3,}:?\s*)+\|?$/.test(s);
}

function splitTableRow(line) {
  const raw = String(line || "").trim();
  const noOuter =
    raw.startsWith("|") || raw.endsWith("|") ? raw.replace(/^\|/, "").replace(/\|$/, "") : raw;
  return noOuter.split("|").map((c) => c.trim());
}

function renderMarkdown(md) {
  const lines = String(md || "").replace(/\r\n/g, "\n").split("\n");
  let html = "";
  let i = 0;

  function renderParagraph(blockLines) {
    const escaped = escapeHTML(blockLines.join("\n"));
    const inline = applyInlineMarkdown(escaped).replace(/\n/g, "<br/>");
    return `<p>${inline}</p>`;
  }

  while (i < lines.length) {
    const line = lines[i];
    if (!line || !line.trim()) {
      i += 1;
      continue;
    }

    const fence = line.trim();
    if (fence.startsWith("```")) {
      const codeLines = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        codeLines.push(lines[i]);
        i += 1;
      }
      if (i < lines.length && lines[i].trim().startsWith("```")) i += 1;
      html += `<pre><code>${escapeHTML(codeLines.join("\n"))}</code></pre>`;
      continue;
    }

    const next = i + 1 < lines.length ? lines[i + 1] : "";
    if (String(line).includes("|") && isTableSeparatorLine(next)) {
      const header = splitTableRow(line);
      const rows = [];
      i += 2;
      while (i < lines.length && lines[i].trim() && String(lines[i]).includes("|")) {
        rows.push(splitTableRow(lines[i]));
        i += 1;
      }
      const thead =
        "<thead><tr>" +
        header.map((c) => `<th>${applyInlineMarkdown(escapeHTML(c))}</th>`).join("") +
        "</tr></thead>";
      const tbody =
        "<tbody>" +
        rows
          .map(
            (r) =>
              "<tr>" + r.map((c) => `<td>${applyInlineMarkdown(escapeHTML(c))}</td>`).join("") + "</tr>"
          )
          .join("") +
        "</tbody>";
      html += `<div class="md-table-wrap"><table>${thead}${tbody}</table></div>`;
      continue;
    }

    const headingMatch = line.match(/^\s{0,3}(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      const level = Math.max(1, Math.min(6, headingMatch[1].length));
      const txt = applyInlineMarkdown(escapeHTML(headingMatch[2] || ""));
      html += `<h${level}>${txt}</h${level}>`;
      i += 1;
      continue;
    }

    const ulMatch = line.match(/^\s*[-*]\s+(.*)$/);
    if (ulMatch) {
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(/^\s*[-*]\s+(.*)$/);
        if (!m) break;
        items.push(`<li>${applyInlineMarkdown(escapeHTML(m[1] || ""))}</li>`);
        i += 1;
      }
      html += `<ul>${items.join("")}</ul>`;
      continue;
    }

    const olMatch = line.match(/^\s*\d+\.\s+(.*)$/);
    if (olMatch) {
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(/^\s*\d+\.\s+(.*)$/);
        if (!m) break;
        items.push(`<li>${applyInlineMarkdown(escapeHTML(m[1] || ""))}</li>`);
        i += 1;
      }
      html += `<ol>${items.join("")}</ol>`;
      continue;
    }

    const block = [];
    while (i < lines.length && lines[i] && lines[i].trim()) {
      block.push(lines[i]);
      i += 1;
    }
    html += renderParagraph(block);
  }
  return html || `<p>${escapeHTML(md)}</p>`;
}

function addChatMessage(role, text) {
  const msg = el("div", `msg ${role}`);
  msg.appendChild(el("div", "role", role === "assistant" ? "Coach" : "You"));
  const bubble = el("div", "bubble");
  bubble.innerHTML = renderMarkdown(text);
  msg.appendChild(bubble);
  if (role === "assistant") {
    const actions = el("div", "actions");
    const speakBtn = el("button", "small-btn", "Speak");
    speakBtn.onclick = () => {
      try {
        const u = new SpeechSynthesisUtterance(text);
        u.rate = 1.0;
        u.pitch = 1.0;
        speechSynthesis.cancel();
        speechSynthesis.speak(u);
      } catch (e) {
        // ignore
      }
    };
    actions.appendChild(speakBtn);
    msg.appendChild(actions);
  }
  ui.chatLog.appendChild(msg);
  ui.chatLog.scrollTop = ui.chatLog.scrollHeight;
}

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let text = await res.text();
    try {
      const j = JSON.parse(text);
      if (j && typeof j.detail === "string") text = j.detail;
    } catch (e) {
      // ignore
    }
    throw new Error(text);
  }
  return await res.json();
}

async function refreshSessions() {
  const sessions = await fetchJSON("/api/sessions");
  ui.sessionSelect.innerHTML = "";
  for (const s of sessions) {
    const opt = document.createElement("option");
    opt.value = s.session_id;
    const shot = s.shot_index ? `shot ${s.shot_index} • ` : "";
    opt.textContent = `${s.session_id} • ${shot}${s.stroke_type || "unknown"} • sim ${
      s.form_similarity !== undefined ? Number(s.form_similarity).toFixed(3) : "—"
    }`;
    ui.sessionSelect.appendChild(opt);
  }
  if (sessions.length > 0) {
    const first = sessions[0].session_id;
    ui.sessionSelect.value = first;
    await loadSession(first);
  }
}

async function loadSession(sessionId) {
  currentSessionId = sessionId;
  sessionData = await fetchJSON(`/api/sessions/${encodeURIComponent(sessionId)}`);

  const sim = sessionData?.metrics?.form_similarity;
  ui.simValue.textContent = sim !== undefined ? Number(sim).toFixed(3) : "—";
  ui.expertValue.textContent = sessionData?.expert?.template_path || "—";

  ui.feedbackList.innerHTML = "";
  const feedback = sessionData?.feedback || [];
  for (const line of feedback) {
    ui.feedbackList.appendChild(el("li", "", line));
  }

  const t = sessionData?.timeline?.t || [];
  ui.frameSlider.max = Math.max(0, t.length - 1);
  ui.frameSlider.value = "0";
  setFrame(0);
  if (viewer && typeof viewer.ensureMeshLoaded === "function") {
    await viewer.ensureMeshLoaded().catch(() => {});
  }

  ui.chatLog.innerHTML = "";
  chatHistory = [];
  addChatMessage(
    "assistant",
    "Ask me anything about your stroke (timing, elbow bend, knees, trunk), and I’ll tie it to the expert template."
  );
}

let anim = { playing: false, handle: null, frame: 0, cursor: 0, last: 0 };

let viewer = null;
let caps = null;

async function loadScript(url) {
  await new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = url;
    s.async = true;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error(`Failed to load ${url}`));
    document.head.appendChild(s);
  });
}

async function ensureThree() {
  if (window.THREE) return;
  try {
    await loadScript("/static/vendor/three.min.js");
    if (window.THREE) return;
  } catch (e) {
    // fall back to CDN
  }

  try {
    await loadScript("https://unpkg.com/three@0.160.0/build/three.min.js");
    if (window.THREE) return;
  } catch (e) {
    // try another CDN
  }

  await loadScript("https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js");
  if (!window.THREE) {
    throw new Error(
      "Failed to load three.js. If your network blocks CDNs, download three.min.js into webapp/static/vendor/three.min.js and reload."
    );
  }
}

async function refreshCapabilities() {
  try {
    caps = await fetchJSON("/api/capabilities");
  } catch (e) {
    caps = null;
  }
  if (ui.genMeshBtn) {
    const ok = Boolean(caps && caps.mesh_generation && caps.mesh_generation.available);
    ui.genMeshBtn.disabled = !ok;
    if (!ok) {
      const hint =
        (caps && caps.mesh_generation && caps.mesh_generation.hint) ||
        "Mesh generation is not configured on the server.";
      setMeshStatus(hint);
    } else {
      setMeshStatus("");
    }
  }
}

function createSimpleOrbitControls(camera, domElement) {
  const state = {
    yaw: 0.0,
    pitch: 0.25,
    distance: 3.5,
    target: new THREE.Vector3(0, 0.9, 0),
    drag: null,
  };

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function update() {
    const cp = Math.cos(state.pitch);
    const sp = Math.sin(state.pitch);
    const cy = Math.cos(state.yaw);
    const sy = Math.sin(state.yaw);
    camera.position.set(
      state.target.x + state.distance * sy * cp,
      state.target.y + state.distance * sp,
      state.target.z + state.distance * cy * cp
    );
    camera.lookAt(state.target);
  }

  function onPointerDown(e) {
    state.drag = {
      x: e.clientX,
      y: e.clientY,
      button: e.button,
      yaw: state.yaw,
      pitch: state.pitch,
      target: state.target.clone(),
    };
    try {
      domElement.setPointerCapture(e.pointerId);
    } catch (err) {
      // ignore
    }
  }

  function onPointerMove(e) {
    if (!state.drag) return;
    const dx = e.clientX - state.drag.x;
    const dy = e.clientY - state.drag.y;
    const isPan = state.drag.button === 2;
    if (isPan) {
      const forward = new THREE.Vector3().subVectors(state.target, camera.position).normalize();
      const upWorld = new THREE.Vector3(0, 1, 0);
      const right = new THREE.Vector3().crossVectors(forward, upWorld).normalize();
      const up = new THREE.Vector3().crossVectors(right, forward).normalize();
      const panSpeed = 0.0022 * state.distance;
      state.target.copy(state.drag.target);
      state.target.addScaledVector(right, -dx * panSpeed);
      state.target.addScaledVector(up, dy * panSpeed);
    } else {
      state.yaw = state.drag.yaw - dx * 0.007;
      state.pitch = clamp(state.drag.pitch - dy * 0.007, -1.2, 1.2);
    }
    update();
  }

  function onPointerUp() {
    state.drag = null;
  }

  function onWheel(e) {
    e.preventDefault();
    const delta = Math.sign(e.deltaY);
    state.distance = clamp(state.distance * (1.0 + delta * 0.08), 1.5, 12.0);
    update();
  }

  domElement.addEventListener("contextmenu", (e) => e.preventDefault());
  domElement.addEventListener("pointerdown", onPointerDown);
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
  domElement.addEventListener("wheel", onWheel, { passive: false });

  update();
  return { target: state.target, update };
}

async function loadMeshVertices(sessionId, fileName) {
  const url = `/sessions/${encodeURIComponent(sessionId)}/${encodeURIComponent(fileName)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load mesh vertices: ${url}`);
  const buf = await res.arrayBuffer();
  return new Float32Array(buf);
}

async function loadUint16Buffer(sessionId, fileName) {
  const url = `/sessions/${encodeURIComponent(sessionId)}/${encodeURIComponent(fileName)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load mesh labels: ${url}`);
  const buf = await res.arrayBuffer();
  return new Uint16Array(buf);
}

function flattenJoints3d(frames) {
  if (!Array.isArray(frames) || frames.length === 0) return null;
  const jointCount = Array.isArray(frames[0]) ? frames[0].length : 0;
  if (!jointCount) return null;
  const out = new Float32Array(frames.length * jointCount * 3);
  let k = 0;
  for (let t = 0; t < frames.length; t++) {
    const f = frames[t];
    if (!Array.isArray(f) || f.length !== jointCount) return null;
    for (let j = 0; j < jointCount; j++) {
      const p = f[j];
      out[k++] = Number((p && p[0]) || 0);
      out[k++] = Number((p && p[1]) || 0);
      out[k++] = Number((p && p[2]) || 0);
    }
  }
  return { flat: out, frameCount: frames.length, jointCount };
}

function normalizePalette(palette) {
  if (!Array.isArray(palette)) return [];
  return palette
    .filter((c) => Array.isArray(c) && c.length === 3)
    .map((c) => {
      let r = Number(c[0]);
      let g = Number(c[1]);
      let b = Number(c[2]);
      if (Math.max(r, g, b) > 1.0) {
        r /= 255.0;
        g /= 255.0;
        b /= 255.0;
      }
      return [r, g, b];
    });
}

function buildVertexColorsFromLabels(labels, palette, vertexCount) {
  const pal = normalizePalette(palette);
  if (!labels || labels.length !== vertexCount || pal.length === 0) return null;
  const colors = new Float32Array(vertexCount * 3);
  for (let i = 0; i < vertexCount; i++) {
    const id = labels[i] % pal.length;
    const c = pal[id];
    colors[i * 3 + 0] = c[0];
    colors[i * 3 + 1] = c[1];
    colors[i * 3 + 2] = c[2];
  }
  return colors;
}

function initMeshObject(color, opacity) {
  const geometry = new THREE.BufferGeometry();
  const material = new THREE.MeshStandardMaterial({
    color,
    transparent: opacity < 1,
    opacity,
    roughness: 0.65,
    metalness: 0.05,
    side: THREE.DoubleSide,
    vertexColors: false,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.visible = false;
  return { mesh, geometry, material, positions: null, colors: null };
}

function initSkeletonObject(color, opacity) {
  const geometry = new THREE.BufferGeometry();
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: opacity < 1,
    opacity,
    vertexColors: false,
  });
  const lines = new THREE.LineSegments(geometry, material);
  lines.visible = false;
  lines.renderOrder = 10;
  return { lines, geometry, material, positions: null };
}

function applyFacesToGeometry(geometry, faces) {
  const idx = new Uint32Array(faces.length * 3);
  for (let i = 0; i < faces.length; i++) {
    idx[i * 3 + 0] = faces[i][0];
    idx[i * 3 + 1] = faces[i][1];
    idx[i * 3 + 2] = faces[i][2];
  }
  geometry.setIndex(new THREE.BufferAttribute(idx, 1));
}

function setMeshFrame(meshObj, verticesFlat, frameIdx, vertexCount) {
  if (!meshObj.positions) return;
  const stride = vertexCount * 3;
  const off = frameIdx * stride;
  const src = verticesFlat.subarray(off, off + stride);
  meshObj.positions.array.set(src);
  meshObj.positions.needsUpdate = true;
  meshObj.geometry.computeVertexNormals();
}

function setSkeletonFrame(skelObj, jointsFlat, frameIdx, jointCount, bones) {
  if (!skelObj.positions) return;
  const stride = jointCount * 3;
  const off = frameIdx * stride;
  const pos = skelObj.positions.array;
  for (let i = 0; i < bones.length; i++) {
    const a = bones[i][0] | 0;
    const b = bones[i][1] | 0;
    const ao = off + a * 3;
    const bo = off + b * 3;
    const p = i * 6;
    pos[p + 0] = jointsFlat[ao + 0];
    pos[p + 1] = jointsFlat[ao + 1];
    pos[p + 2] = jointsFlat[ao + 2];
    pos[p + 3] = jointsFlat[bo + 0];
    pos[p + 4] = jointsFlat[bo + 1];
    pos[p + 5] = jointsFlat[bo + 2];
  }
  skelObj.positions.needsUpdate = true;
}

function updateMeshDiffColors(meshObj, userVerts, expertVerts, frameIdx, vertexCount) {
  if (!meshObj.colors) return;
  const stride = vertexCount * 3;
  const off = frameIdx * stride;
  const u = userVerts.subarray(off, off + stride);
  const e = expertVerts.subarray(off, off + stride);
  const c = meshObj.colors.array;
  for (let i = 0; i < vertexCount; i++) {
    const dx = u[i * 3 + 0] - e[i * 3 + 0];
    const dy = u[i * 3 + 1] - e[i * 3 + 1];
    const dz = u[i * 3 + 2] - e[i * 3 + 2];
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const t = Math.max(0, Math.min(1, d / 0.25));
    const r = (90 + (255 - 90) * t) / 255;
    const g = (167 + (92 - 167) * t) / 255;
    const b = (255 + (122 - 255) * t) / 255;
    c[i * 3 + 0] = r;
    c[i * 3 + 1] = g;
    c[i * 3 + 2] = b;
  }
  meshObj.colors.needsUpdate = true;
}

async function init3D() {
  const canvas = document.getElementById("viewer");
  await ensureThree();

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x0b1020, 6, 16);

  const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 100);
  camera.position.set(0.0, 1.4, 3.5);

  const controls = createSimpleOrbitControls(camera, canvas);

  scene.add(new THREE.HemisphereLight(0xbfd7ff, 0x0b1020, 0.85));
  const dir = new THREE.DirectionalLight(0xffffff, 0.9);
  dir.position.set(2, 4, 2);
  scene.add(dir);

  const grid = new THREE.GridHelper(10, 20, 0x335599, 0x1e2a55);
  grid.position.y = 0;
  grid.material.opacity = 0.25;
  grid.material.transparent = true;
  scene.add(grid);

  const userMesh = initMeshObject(0x5aa7ff, 0.95);
  const expertMesh = initMeshObject(0xffb86b, 0.70);
  scene.add(userMesh.mesh);
  scene.add(expertMesh.mesh);

  const userSkel = initSkeletonObject(0x5aa7ff, 0.95);
  const expertSkel = initSkeletonObject(0xffb86b, 0.75);
  scene.add(userSkel.lines);
  scene.add(expertSkel.lines);

  const state = {
    renderer,
    scene,
    camera,
    controls,
    userMesh,
    expertMesh,
    userSkel,
    expertSkel,
    sessionId: null,
    poseKey: null,
    bones: DEFAULT_COCO17_BONES,
    jointCount: 0,
    poseFrameCount: 0,
    userJoints: null,
    expertJoints: null,
    meshKey: null,
    faces: null,
    vertexCount: 0,
    frameCount: 0,
    userVerts: null,
    expertVerts: null,
    vertexLabels: null,
    partColors: null,
    colorMode: "solid",
    lastFrame: -1,
    dirty: true,
  };
  viewer = state;

  function resize() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function ensurePoseLoaded() {
    if (!sessionData || !currentSessionId) return;

    const bones = sessionData?.joints?.bones;
    const names = sessionData?.joints?.names;
    const userFrames = sessionData?.user?.joints3d_resampled;
    const expertFrames = sessionData?.expert?.joints3d_resampled;
    const poseKey = JSON.stringify({
      sessionId: currentSessionId,
      boneCount: Array.isArray(bones) ? bones.length : null,
      jointCount: Array.isArray(names) ? names.length : null,
      userFrames: Array.isArray(userFrames) ? userFrames.length : null,
      expertFrames: Array.isArray(expertFrames) ? expertFrames.length : null,
    });
    if (state.poseKey === poseKey) return;

    state.poseKey = poseKey;
    state.bones = Array.isArray(bones) && bones.length ? bones : DEFAULT_COCO17_BONES;

    const userFlat = flattenJoints3d(userFrames);
    const expertFlat = flattenJoints3d(expertFrames);
    state.userJoints = userFlat ? userFlat.flat : null;
    state.expertJoints = expertFlat ? expertFlat.flat : null;
    state.jointCount = userFlat ? userFlat.jointCount : expertFlat ? expertFlat.jointCount : 0;
    state.poseFrameCount = userFlat ? userFlat.frameCount : expertFlat ? expertFlat.frameCount : 0;

    const segCount = state.bones.length;
    userSkel.positions = new THREE.BufferAttribute(new Float32Array(segCount * 2 * 3), 3);
    userSkel.geometry.setAttribute("position", userSkel.positions);
    expertSkel.positions = new THREE.BufferAttribute(new Float32Array(segCount * 2 * 3), 3);
    expertSkel.geometry.setAttribute("position", expertSkel.positions);
    state.dirty = true;
  }

  async function ensureMeshLoaded() {
    if (!sessionData || !currentSessionId) return;
    ensurePoseLoaded();

    const faces = sessionData?.mesh?.faces;
    const userPath = sessionData?.user?.mesh_vertices_path;
    const expertPath = sessionData?.expert?.mesh_vertices_path;
    const vertexCount = sessionData?.mesh?.vertex_count;
    const frameCount = sessionData?.mesh?.frame_count;
    const faceCount =
      sessionData?.mesh?.face_count !== undefined
        ? Number(sessionData.mesh.face_count)
        : Array.isArray(faces)
          ? faces.length
          : 0;
    const labelsPath = sessionData?.mesh?.vertex_labels_path;
    const palette = sessionData?.mesh?.parts_palette;
    const meshKey = JSON.stringify({
      sessionId: currentSessionId,
      userPath: userPath || null,
      expertPath: expertPath || null,
      vertexCount: vertexCount || null,
      frameCount: frameCount || null,
      faceCount: faceCount || null,
      labelsPath: labelsPath || null,
      paletteLen: Array.isArray(palette) ? palette.length : null,
    });
    if (state.meshKey === meshKey) return;

    state.sessionId = currentSessionId;
    state.meshKey = meshKey;
    state.faces = null;
    state.vertexCount = 0;
    state.frameCount = 0;
    state.userVerts = null;
    state.expertVerts = null;
    state.vertexLabels = null;
    state.partColors = null;
    state.colorMode = "solid";
    state.lastFrame = -1;

    if (!faces || !userPath || !vertexCount || !frameCount) {
      userMesh.mesh.visible = false;
      expertMesh.mesh.visible = false;
      const hasPose = Boolean(state.userJoints && state.bones && state.bones.length);
      if (ui.viewerNotice) ui.viewerNotice.classList.toggle("hidden", hasPose);
      state.dirty = true;
      return;
    }

    state.faces = faces;
    state.vertexCount = vertexCount;
    state.frameCount = frameCount;

    applyFacesToGeometry(userMesh.geometry, faces);
    applyFacesToGeometry(expertMesh.geometry, faces);

    userMesh.positions = new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3);
    userMesh.geometry.setAttribute("position", userMesh.positions);
    userMesh.colors = new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3);
    userMesh.geometry.setAttribute("color", userMesh.colors);

    expertMesh.positions = new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3);
    expertMesh.geometry.setAttribute("position", expertMesh.positions);
    expertMesh.colors = new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3);
    expertMesh.geometry.setAttribute("color", expertMesh.colors);

    state.userVerts = await loadMeshVertices(currentSessionId, userPath);
    if (expertPath) {
      try {
        state.expertVerts = await loadMeshVertices(currentSessionId, expertPath);
      } catch (e) {
        state.expertVerts = null;
      }
    }

    if (labelsPath && palette) {
      try {
        state.vertexLabels = await loadUint16Buffer(currentSessionId, labelsPath);
        state.partColors = buildVertexColorsFromLabels(state.vertexLabels, palette, vertexCount);
      } catch (e) {
        state.vertexLabels = null;
        state.partColors = null;
      }
    }

    userMesh.mesh.visible = true;
    expertMesh.mesh.visible = Boolean(state.expertVerts);
    if (ui.viewerNotice) ui.viewerNotice.classList.add("hidden");
    state.dirty = true;
  }
  state.ensureMeshLoaded = ensureMeshLoaded;

  function updateFrame() {
    const idx = anim.frame || 0;
    if (!sessionData) return;
    ensurePoseLoaded();

    const mode = ui.viewMode.value;
    const ghost = ui.ghostToggle.checked;

    const userOffset = mode === "side_by_side" ? -0.9 : 0.0;
    const expertOffset = mode === "side_by_side" ? 0.9 : 0.0;
    userMesh.mesh.position.set(userOffset, 0, 0);
    expertMesh.mesh.position.set(expertOffset, 0, 0);
    userSkel.lines.position.set(userOffset, 0, 0);
    expertSkel.lines.position.set(expertOffset, 0, 0);

    const hasMesh = Boolean(state.userVerts && state.faces);
    userMesh.mesh.visible = hasMesh;
    expertMesh.mesh.visible = Boolean(hasMesh && state.expertVerts && ghost);
    userSkel.lines.visible = false;
    expertSkel.lines.visible = false;

    if (!hasMesh) {
      expertMesh.mesh.visible = false;
      const hasPose = Boolean(state.userJoints && state.bones && state.bones.length);
      userSkel.lines.visible = Boolean(hasPose);
      expertSkel.lines.visible = Boolean(hasPose && state.expertJoints && ghost);
      if (ui.viewerNotice) ui.viewerNotice.classList.toggle("hidden", hasPose);
      if (!hasPose) return;
      const f = Math.max(0, Math.min(state.poseFrameCount - 1, idx));
      setSkeletonFrame(userSkel, state.userJoints, f, state.jointCount, state.bones);
      if (state.expertJoints && ghost) {
        setSkeletonFrame(expertSkel, state.expertJoints, f, state.jointCount, state.bones);
      }
      return;
    }

    if (ui.viewerNotice) ui.viewerNotice.classList.add("hidden");

    const f = Math.max(0, Math.min(state.frameCount - 1, idx));
    setMeshFrame(userMesh, state.userVerts, f, state.vertexCount);
    if (state.expertVerts && ghost) {
      setMeshFrame(expertMesh, state.expertVerts, f, state.vertexCount);
    }
    const diffOn = ui.diffToggle.checked && state.expertVerts && ghost && mode === "overlay";
    const partsOn = ui.partsToggle.checked && Boolean(state.partColors) && !diffOn;
    const desiredMode = diffOn ? "diff" : partsOn ? "parts" : "solid";

    if (state.colorMode !== desiredMode) {
      state.colorMode = desiredMode;
      if (desiredMode === "solid") {
        userMesh.material.vertexColors = false;
        expertMesh.material.vertexColors = false;
      } else if (desiredMode === "parts") {
        userMesh.material.vertexColors = true;
        expertMesh.material.vertexColors = true;
        if (state.partColors) {
          userMesh.colors.array.set(state.partColors);
          userMesh.colors.needsUpdate = true;
          if (expertMesh.colors) {
            expertMesh.colors.array.set(state.partColors);
            expertMesh.colors.needsUpdate = true;
          }
        }
      } else if (desiredMode === "diff") {
        userMesh.material.vertexColors = true;
        expertMesh.material.vertexColors = false;
      }
    }

    if (diffOn) {
      updateMeshDiffColors(userMesh, state.userVerts, state.expertVerts, f, state.vertexCount);
    }
  }

  function loop() {
    requestAnimationFrame(loop);
    resize();
    state.controls.update();
    if (state.dirty || anim.frame !== state.lastFrame) {
      state.lastFrame = anim.frame;
      state.dirty = false;
      updateFrame();
    }
    state.renderer.render(state.scene, state.camera);
  }

  window.addEventListener("resize", resize);
  loop();

  // Keep mesh loading opportunistic (triggered by session switches).
  setInterval(() => ensureMeshLoaded().catch(() => {}), 1500);
}

async function pollJob(jobId) {
  const url = `/api/jobs/${encodeURIComponent(jobId)}`;
  while (true) {
    const st = await fetchJSON(url);
    const status = st.status || "unknown";
    const detail = st.detail ? ` • ${st.detail}` : "";
    setMeshStatus(`${status}${detail}`);
    if (status === "done") return;
    if (status === "error") throw new Error(st.error || "Mesh generation failed.");
    await new Promise((r) => setTimeout(r, 1000));
  }
}

async function generateMesh() {
  if (!currentSessionId) {
    setMeshStatus("Select a session first.");
    return;
  }
  if (!ui.genMeshBtn) return;
  ui.genMeshBtn.disabled = true;
  setMeshStatus("Starting mesh generation…");
  try {
    const res = await fetchJSON(
      `/api/sessions/${encodeURIComponent(currentSessionId)}/mesh/generate`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          include_expert: Boolean(ui.genMeshExpert && ui.genMeshExpert.checked),
          clip_to_window: Boolean(ui.genMeshClip && ui.genMeshClip.checked),
        }),
      }
    );
    const jobId = res.job_id;
    if (!jobId) throw new Error("Server did not return job_id.");
    await pollJob(jobId);
    setMeshStatus("done • mesh attached");
    await loadSession(currentSessionId);
  } catch (e) {
    setMeshStatus(`error • ${String(e.message || e)}`);
  } finally {
    ui.genMeshBtn.disabled = false;
  }
}

function setFrame(frameIdx) {
  if (!sessionData) return;
  const t = sessionData?.timeline?.t || [];
  const n = t.length;
  const idx = Math.max(0, Math.min(n - 1, frameIdx));
  anim.frame = idx;
  anim.cursor = idx;
  if (viewer) viewer.dirty = true;
  ui.frameSlider.value = String(idx);
  ui.frameLabel.textContent = `frame ${idx}`;
  const tVal = t[idx] !== undefined ? t[idx] : 0;
  ui.timeLabel.textContent = `t=${Number(tVal).toFixed(2)}`;
}

function play() {
  if (!sessionData) return;
  anim.playing = true;
  anim.last = performance.now();
  anim.cursor = anim.frame || 0;
  tick();
}

function pause() {
  anim.playing = false;
  if (anim.handle) cancelAnimationFrame(anim.handle);
  anim.handle = null;
}

function tick() {
  if (!anim.playing) return;
  const now = performance.now();
  const dt = now - anim.last;
  anim.last = now;

  const speed = Number(ui.speedSelect.value || "1");
  const framesPerSecond = 30;
  const step = (dt / 1000) * framesPerSecond * speed;
  let next = anim.cursor + step;
  const n = sessionData?.timeline?.t?.length || 0;
  if (n > 0 && next >= n) next = 0;
  anim.cursor = next;
  setFrame(Math.floor(anim.cursor));

  anim.handle = requestAnimationFrame(tick);
}

async function sendChat() {
  const text = ui.chatText.value.trim();
  if (!text) return;
  if (!currentSessionId) {
    setError("Create or select a session first.");
    return;
  }
  setError("");
  ui.chatText.value = "";
  addChatMessage("user", text);
  chatHistory.push({ role: "user", content: text });

  try {
    const res = await fetchJSON("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: currentSessionId, messages: chatHistory.slice(-16) }),
    });
    const answer = res.answer || "(no response)";
    addChatMessage("assistant", answer);
    chatHistory.push({ role: "assistant", content: answer });
  } catch (e) {
    setError(String(e.message || e));
    addChatMessage("assistant", "I couldn't reach the coaching model. Check server logs and LLM env vars.");
  }
}

function openModal() {
  ui.createStatus.textContent = "";
  ui.modal.classList.remove("hidden");
}
function closeModal() {
  ui.modal.classList.add("hidden");
}

async function createSessionFromModal() {
  ui.createStatus.textContent = "Building session… (this can take a bit)";
  const payload = {
    video_path: ui.videoPath.value.trim(),
    t_impact: ui.tImpact.value ? Number(ui.tImpact.value) : null,
    stroke_type: ui.strokeType.value.trim() || null,
    multi_shot: Boolean(ui.multiShot.checked) && !ui.tImpact.value,
    max_shots: Number(ui.maxShots.value || "5"),
    min_shot_separation_s: Number(ui.minSep.value || "1.0"),
    view: ui.viewFilter.value || null,
    resample_len: Number(ui.resampleLen.value || "120"),
  };
  try {
    const res = await fetchJSON("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    ui.createStatus.textContent = "Session created.";
    closeModal();
    await refreshSessions();
    if (res.session_id) {
      ui.sessionSelect.value = res.session_id;
      await loadSession(res.session_id);
    } else if (Array.isArray(res.sessions) && res.sessions.length > 0) {
      const first = res.sessions[0].session_id;
      ui.sessionSelect.value = first;
      await loadSession(first);
    }
  } catch (e) {
    ui.createStatus.textContent = `Failed: ${String(e.message || e)}`;
  }
}

function wireUI() {
  ui.sessionSelect.onchange = async () => {
    await loadSession(ui.sessionSelect.value);
  };
  ui.frameSlider.oninput = () => {
    pause();
    setFrame(Number(ui.frameSlider.value));
  };
  ui.playBtn.onclick = () => play();
  ui.pauseBtn.onclick = () => pause();
  ui.speedSelect.onchange = () => {};
  ui.viewMode.onchange = () => setFrame(anim.frame);
  ui.ghostToggle.onchange = () => setFrame(anim.frame);
  ui.partsToggle.onchange = () => setFrame(anim.frame);
  ui.diffToggle.onchange = () => setFrame(anim.frame);

  ui.sendBtn.onclick = () => sendChat();
  ui.chatText.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendChat();
  });

  ui.newSessionBtn.onclick = () => openModal();
  ui.cancelBtn.onclick = () => closeModal();
  ui.createBtn.onclick = () => createSessionFromModal();

  if (ui.genMeshBtn) {
    ui.genMeshBtn.onclick = () => generateMesh();
  }

  ui.modal.addEventListener("click", (e) => {
    if (e.target === ui.modal) closeModal();
  });
}

function initResizablePanes() {
  const layout = document.querySelector(".layout");
  const leftPane = document.querySelector(".panel.left");
  const rightPane = document.querySelector(".panel.right");
  const leftHandle = ui.leftResizer;
  const rightHandle = ui.rightResizer;
  if (!layout || !leftPane || !rightPane || !leftHandle || !rightHandle) return;

  const minLeft = 260;
  const minRight = 300;
  const minCenter = 360;
  const splitterW = Math.max(6, leftHandle.getBoundingClientRect().width || 12);

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function setLeft(w) {
    layout.style.setProperty("--left-pane-w", `${Math.round(w)}px`);
  }
  function setRight(w) {
    layout.style.setProperty("--right-pane-w", `${Math.round(w)}px`);
  }

  const storedLeft = Number(localStorage.getItem("tcvision_leftPaneW") || "");
  const storedRight = Number(localStorage.getItem("tcvision_rightPaneW") || "");
  if (Number.isFinite(storedLeft) && storedLeft > 0) setLeft(storedLeft);
  if (Number.isFinite(storedRight) && storedRight > 0) setRight(storedRight);

  function beginDrag(which, ev) {
    ev.preventDefault();
    const startX = ev.clientX;
    const startLeft = leftPane.getBoundingClientRect().width;
    const startRight = rightPane.getBoundingClientRect().width;
    document.body.classList.add("resizing");

    try {
      (which === "left" ? leftHandle : rightHandle).setPointerCapture(ev.pointerId);
    } catch (e) {
      // ignore
    }

    function onMove(e) {
      const dx = e.clientX - startX;
      const totalW = layout.getBoundingClientRect().width;

      if (which === "left") {
        const maxLeft = totalW - startRight - splitterW * 2 - minCenter;
        const next = clamp(startLeft + dx, minLeft, Math.max(minLeft, maxLeft));
        setLeft(next);
      } else {
        const maxRight = totalW - startLeft - splitterW * 2 - minCenter;
        const next = clamp(startRight - dx, minRight, Math.max(minRight, maxRight));
        setRight(next);
      }
    }

    function onUp() {
      document.body.classList.remove("resizing");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      localStorage.setItem("tcvision_leftPaneW", String(Math.round(leftPane.getBoundingClientRect().width)));
      localStorage.setItem("tcvision_rightPaneW", String(Math.round(rightPane.getBoundingClientRect().width)));
    }

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp, { once: true });
  }

  leftHandle.addEventListener("pointerdown", (e) => beginDrag("left", e));
  rightHandle.addEventListener("pointerdown", (e) => beginDrag("right", e));
}

async function main() {
  try {
    await init3D();
  } catch (e) {
    setError(String(e.message || e));
  }
  wireUI();
  initResizablePanes();
  try {
    await refreshSessions();
  } catch (e) {
    setError(`Failed to load sessions: ${String(e.message || e)}`);
  }
}

main();
