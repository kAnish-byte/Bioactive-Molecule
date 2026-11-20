/* ============================================================
   BioPredict — script.js
   Inference, 3D viewer, batch screening, benchmark + UI motion
   ============================================================ */

const API_BASE = (window.location.protocol === "file:")
  ? "http://127.0.0.1:8000"
  : window.location.origin;

const EXAMPLES = [
  "Aspirin", "Celecoxib", "Ibuprofen", "Paracetamol", "Diclofenac", "Naproxen", "Meloxicam", "Caffeine",
  "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
  "CC(=O)Oc1ccccc1C(=O)O",
  "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
  "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
  "O=S(=O)(c1ccc(C2=C(c3ccccc3)C(=O)OC2)cc1)C",
  "Cc1nc(NC(=O)c2c(O)c3ccccc3s(=O)(=O)n2C)cs1",
  "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
  "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
];

const $ = (id) => document.getElementById(id);
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let exampleIdx = 0;
let viewer3D = null;
let isSpinning = true;
let currentStyle = "ballstick";
let currentHighlight = null;
let currentMol3D = "";
let currentAttribution = null;
let isHighlightActive = false;
let batchDataCache = null;
let modelsCache = null;

function startApp() {
  refreshIcons();
  initNav();
  initAuthNav();
  initReveal();
  initCountUp();
  initHealthCheck();
  initHeroDynamicMolecules();
  initSinglePrediction();
  initBatchPrediction();
  initBenchmark();
  initLightbox();
  initChatbot();
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", startApp);
else startApp();

setInterval(initHealthCheck, 10000);

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.9 } });
}

function setButtonContent(btn, icon, label) {
  btn.innerHTML = `<i data-lucide="${icon}"></i> <span>${label}</span>`;
  refreshIcons();
}

function toast(msg) {
  const el = $("toast");
  if (!el) return;
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove("show"), 2200);
}

/* ─── Ambient drifting molecules (optimized lightweight canvas) ─── */
function initAmbientMolecules() {
  const canvas = $("moleculeCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const palette = ["15,148,136", "90,169,230", "233,162,59", "124,196,184"];
  let w, h, molecules = [];
  let lastTime = 0;

  function makeMolecule() {
    const atoms = [];
    const n = 3 + Math.floor(Math.random() * 3);
    const ring = Math.random() > 0.5;
    const r = 12 + Math.random() * 8;
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      atoms.push(ring
        ? { x: Math.cos(a) * r, y: Math.sin(a) * r, s: 2.5 + Math.random() * 1.5 }
        : { x: (i - n / 2) * r * 0.9, y: (i % 2 ? 1 : -1) * r * 0.4, s: 2.5 + Math.random() * 1.5 });
    }
    return {
      x: Math.random() * w, y: Math.random() * h, atoms, ring,
      vx: (Math.random() - 0.5) * 0.16, vy: (Math.random() - 0.5) * 0.16,
      rot: Math.random() * Math.PI, vr: (Math.random() - 0.5) * 0.002,
      color: palette[Math.floor(Math.random() * palette.length)],
      alpha: 0.12 + Math.random() * 0.1, scale: 0.7 + Math.random() * 0.6
    };
  }

  function resize() {
    w = window.innerWidth; h = window.innerHeight;
    canvas.width = w; canvas.height = h;
    canvas.style.width = w + "px"; canvas.style.height = h + "px";
    const count = Math.min(8, Math.max(4, Math.round((w * h) / 120000)));
    molecules = Array.from({ length: count }, makeMolecule);
  }

  function draw(now) {
    if (document.hidden || prefersReducedMotion) {
      requestAnimationFrame(draw);
      return;
    }
    // Throttle to 30 FPS to save CPU and GPU for WebGL 3Dmol
    if (now - lastTime < 33) {
      requestAnimationFrame(draw);
      return;
    }
    lastTime = now;

    ctx.clearRect(0, 0, w, h);
    for (let j = 0; j < molecules.length; j++) {
      const m = molecules[j];
      m.x += m.vx; m.y += m.vy; m.rot += m.vr;
      if (m.x < -60) m.x = w + 60; if (m.x > w + 60) m.x = -60;
      if (m.y < -60) m.y = h + 60; if (m.y > h + 60) m.y = -60;
      ctx.save();
      ctx.translate(m.x, m.y); ctx.rotate(m.rot); ctx.scale(m.scale, m.scale);
      ctx.strokeStyle = `rgba(${m.color},${m.alpha * 0.8})`;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (let i = 0; i < m.atoms.length; i++) {
        const a = m.atoms[i];
        const b = m.atoms[(i + 1) % m.atoms.length];
        if (!m.ring && i === m.atoms.length - 1) continue;
        ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
      }
      ctx.stroke();
      for (let i = 0; i < m.atoms.length; i++) {
        const a = m.atoms[i];
        ctx.beginPath();
        ctx.fillStyle = `rgba(${m.color},${m.alpha + 0.1})`;
        ctx.arc(a.x, a.y, a.s, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }
    requestAnimationFrame(draw);
  }

  resize();
  window.addEventListener("resize", resize, { passive: true });
  requestAnimationFrame(draw);
}

/* ─── Navigation, scroll progress, active links ───────────── */
function initNav() {
  const navbar = $("navbar");
  const progress = $("scrollProgress");
  const toggle = $("navToggle");
  const links = $("navLinks");

  const onScroll = () => {
    const y = window.scrollY;
    navbar?.classList.toggle("scrolled", y > 12);
    const max = document.documentElement.scrollHeight - window.innerHeight;
    if (progress) progress.style.width = `${max > 0 ? (y / max) * 100 : 0}%`;
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  toggle?.addEventListener("click", () => {
    const open = links.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
  });
  links?.querySelectorAll("a").forEach(a => a.addEventListener("click", () => {
    links.classList.remove("open");
    toggle?.setAttribute("aria-expanded", "false");
  }));

  // Resolve active page link from URL pathname
  const currentPath = window.location.pathname.replace(/\/+$/, "") || "/app";
  const navAnchors = links?.querySelectorAll("a");
  if (navAnchors && navAnchors.length) {
    let matched = false;
    navAnchors.forEach(a => {
      const page = a.dataset.page;
      const href = (a.getAttribute("href") || "").replace(/\/+$/, "");
      const isHome = (currentPath === "/app" || currentPath === "" || currentPath.endsWith("/index.html")) &&
                     (page === "home" || href === "/app" || href === "/app/index.html");
      const isOtherMatch = page && (
        (page === "predict" && currentPath.endsWith("/predict")) ||
        (page === "batch" && currentPath.endsWith("/batch")) ||
        (page === "benchmark" && currentPath.endsWith("/benchmark")) ||
        (page === "bioactivity" && (currentPath.endsWith("/bioactivity-guide") || currentPath.endsWith("/bioactivity") || currentPath.endsWith("/bioactivity.html"))) ||
        (page === "about" && currentPath.endsWith("/about"))
      );
      if (isHome || isOtherMatch || (href && currentPath === href)) {
        a.classList.add("active");
        a.setAttribute("aria-current", "page");
        matched = true;
      } else {
        a.classList.remove("active");
        a.removeAttribute("aria-current");
      }
    });
    if (!matched && (currentPath === "/app" || currentPath === "")) {
      const homeLink = links.querySelector('[data-page="home"]') || navAnchors[0];
      if (homeLink) {
        homeLink.classList.add("active");
        homeLink.setAttribute("aria-current", "page");
      }
    }
  }
}

/* ─── Authentication state & Navbar User Chip ──────────────── */
function initAuthNav() {
  const navRight = document.querySelector(".nav-right");
  if (!navRight) return;

  // Look for existing slot or create one
  let slot = $("navAuthSlot");
  if (!slot) {
    slot = document.createElement("div");
    slot.id = "navAuthSlot";
    slot.className = "nav-auth-slot";
    const toggle = $("navToggle");
    if (toggle) navRight.insertBefore(slot, toggle);
    else navRight.appendChild(slot);
  }

  const token = localStorage.getItem("biopredict_token");
  let user = null;
  try {
    const raw = localStorage.getItem("biopredict_user");
    if (raw) user = JSON.parse(raw);
  } catch {
    user = null;
  }

  if (token && user) {
    const initial = (user.username || "U").charAt(0).toUpperCase();
    slot.innerHTML = `
      <div class="user-chip-wrap" id="userChipWrap">
        <button type="button" class="user-chip" id="userChipBtn" aria-expanded="false" title="Account: ${user.username}">
          <span class="user-avatar-circle">${initial}</span>
          <span class="user-chip-name">${user.username}</span>
          <i data-lucide="chevron-down"></i>
        </button>
        <div class="user-dropdown" id="userDropdown" hidden>
          <div class="user-drop-head">
            <strong>${user.username}</strong>
            <span class="user-drop-email">${user.email || 'Researcher'}</span>
          </div>
          <div class="user-drop-divider"></div>
          <a href="/app/predict" class="user-drop-item"><i data-lucide="flask-conical"></i> Screening Workspace</a>
          <button type="button" class="user-drop-item" id="btnLogout"><i data-lucide="log-out"></i> Sign Out</button>
        </div>
      </div>
    `;

    const wrap = $("userChipWrap");
    const btn = $("userChipBtn");
    const dropdown = $("userDropdown");
    const btnLogout = $("btnLogout");

    btn?.addEventListener("click", (e) => {
      e.stopPropagation();
      const isOpen = !dropdown.hidden;
      dropdown.hidden = isOpen;
      wrap?.classList.toggle("open", !isOpen);
      btn.setAttribute("aria-expanded", String(!isOpen));
    });

    document.addEventListener("click", (e) => {
      if (dropdown && !dropdown.hidden && wrap && !wrap.contains(e.target)) {
        dropdown.hidden = true;
        wrap?.classList.remove("open");
        btn?.setAttribute("aria-expanded", "false");
      }
    });

    btnLogout?.addEventListener("click", () => {
      localStorage.removeItem("biopredict_token");
      localStorage.removeItem("biopredict_user");
      toast("Signed out successfully");
      setTimeout(() => {
        window.location.href = "/app/login?force=1";
      }, 450);
    });
  } else {
    // If not logged in and not on login page, show Sign In button
    if (!window.location.pathname.includes("login")) {
      slot.innerHTML = `
        <a href="/app/login" class="nav-auth-btn" id="navLoginBtn">
          <i data-lucide="user"></i>
          <span>Sign In</span>
        </a>
      `;
    } else {
      slot.innerHTML = "";
    }
  }

  if (window.lucide && typeof lucide.createIcons === "function") {
    lucide.createIcons();
  }
}

function initReveal() {
  const els = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) { els.forEach(e => e.classList.add("in")); return; }
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
  }, { threshold: 0.05, rootMargin: "60px" });
  els.forEach(e => io.observe(e));
  requestAnimationFrame(() => {
    els.forEach(e => {
      const rect = e.getBoundingClientRect();
      if (rect.top < window.innerHeight + 100) e.classList.add("in");
    });
  });
}

function animateNumber(el, target, decimals = 0, duration = 1400, suffix = "") {
  if (!el || typeof el.textContent === "undefined") return;
  const numTarget = Number(target) || 0;
  const start = performance.now();
  const step = (now) => {
    if (!el || typeof el.textContent === "undefined") return;
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = (numTarget * eased).toLocaleString(undefined, {
      minimumFractionDigits: decimals, maximumFractionDigits: decimals
    }) + suffix;
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function initCountUp() {
  const els = document.querySelectorAll(".count-up");
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      const el = e.target;
      animateNumber(el, parseFloat(el.dataset.target), parseInt(el.dataset.decimals || "0", 10));
      io.unobserve(el);
    });
  }, { threshold: 0.5 });
  els.forEach(e => io.observe(e));
}

/* ─── 3Dmol helpers ───────────────────────────────────────── */
function when3DmolReady(cb, attempts = 0) {
  if (window.$3Dmol) return cb(true);
  if (attempts > 40) return cb(false);
  setTimeout(() => when3DmolReady(cb, attempts + 1), 150);
}

function styleFor(kind) {
  if (kind === "stick") return { stick: { radius: 0.16 } };
  return { stick: { radius: 0.13 }, sphere: { scale: 0.24 } };
}

let isViewer3DVisible = true;

/* ─── Hero Dynamic Molecular Simulation (Pure 2D Canvas) ──── */
function initHeroDynamicMolecules() {
  const canvas = $("heroMolCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  let animId = null;
  let isVisible = true;
  let lastTime = 0;
  const fpsInterval = 1000 / 35; // Throttled to 35 FPS for smooth animation with ~0% GPU load

  // Representation of a multi-ring pharmacophore backbone undergoing conformational dynamics
  const atoms = [
    // Core Pyrazole / Central Ring
    { id: 0, sym: "N", x: -20, y: -15, z: 10, color: "#6366f1", r: 8 },
    { id: 1, sym: "N", x: 15, y: -25, z: -5, color: "#4f46e5", r: 8 },
    { id: 2, sym: "C", x: 40, y: 0, z: -15, color: "#0ea5e9", r: 7 },
    { id: 3, sym: "C", x: 20, y: 25, z: -5, color: "#0ea5e9", r: 7 },
    { id: 4, sym: "C", x: -18, y: 15, z: 12, color: "#0ea5e9", r: 7 },
    // Phenyl Ring A (Right wing)
    { id: 5, sym: "C", x: 80, y: -5, z: -30, color: "#0284c7", r: 6.5 },
    { id: 6, sym: "C", x: 110, y: 15, z: -40, color: "#0284c7", r: 6.5 },
    { id: 7, sym: "C", x: 140, y: 10, z: -45, color: "#0284c7", r: 6.5 },
    { id: 8, sym: "C", x: 145, y: -20, z: -40, color: "#0284c7", r: 6.5 },
    { id: 9, sym: "C", x: 118, y: -40, z: -30, color: "#0284c7", r: 6.5 },
    { id: 10, sym: "C", x: 85, y: -32, z: -25, color: "#0284c7", r: 6.5 },
    // Sulfonamide group off ring A (Active Pharmacophore Anchor)
    { id: 11, sym: "S", x: 180, y: 35, z: -50, color: "#f59e0b", r: 9 },
    { id: 12, sym: "O", x: 205, y: 25, z: -30, color: "#ef4444", r: 7 },
    { id: 13, sym: "O", x: 175, y: 65, z: -60, color: "#ef4444", r: 7 },
    { id: 14, sym: "N", x: 195, y: 30, z: -80, color: "#4f46e5", r: 7.5 },
    // Phenyl Ring B (Bottom-Left wing)
    { id: 15, sym: "C", x: -45, y: 38, z: 25, color: "#0284c7", r: 6.5 },
    { id: 16, sym: "C", x: -55, y: 70, z: 30, color: "#0284c7", r: 6.5 },
    { id: 17, sym: "C", x: -85, y: 85, z: 40, color: "#0284c7", r: 6.5 },
    { id: 18, sym: "C", x: -110, y: 68, z: 45, color: "#0284c7", r: 6.5 },
    { id: 19, sym: "C", x: -102, y: 35, z: 40, color: "#0284c7", r: 6.5 },
    { id: 20, sym: "C", x: -70, y: 20, z: 30, color: "#0284c7", r: 6.5 },
    // Halo-substituent cluster (Top-Left)
    { id: 21, sym: "C", x: -40, y: -38, z: 20, color: "#0ea5e9", r: 7 },
    { id: 22, sym: "F", x: -65, y: -30, z: 30, color: "#10b981", r: 6.5 },
    { id: 23, sym: "F", x: -35, y: -65, z: 25, color: "#10b981", r: 6.5 },
    { id: 24, sym: "F", x: -50, y: -45, z: -5, color: "#10b981", r: 6.5 }
  ];

  const bonds = [
    // Central ring
    [0, 1, 1], [1, 2, 2], [2, 3, 1], [3, 4, 2], [4, 0, 1],
    // Right phenyl ring
    [2, 5, 1], [5, 6, 2], [6, 7, 1], [7, 8, 2], [8, 9, 1], [9, 10, 2], [10, 5, 1],
    // Sulfonamide
    [7, 11, 1], [11, 12, 2], [11, 13, 2], [11, 14, 1],
    // Left phenyl ring
    [4, 15, 1], [15, 16, 2], [16, 17, 1], [17, 18, 2], [18, 19, 1], [19, 20, 2], [20, 15, 1],
    // Halo-substituents
    [0, 21, 1], [21, 22, 1], [21, 23, 1], [21, 24, 1]
  ];

  // Floating ambient background probe particles
  const ambientParticles = Array.from({ length: 18 }, () => ({
    x: (Math.random() - 0.5) * 360,
    y: (Math.random() - 0.5) * 280,
    vx: (Math.random() - 0.5) * 0.3,
    vy: (Math.random() - 0.5) * 0.3,
    r: Math.random() * 2.5 + 2,
    color: Math.random() > 0.5 ? "rgba(14, 165, 233, 0.4)" : "rgba(16, 185, 129, 0.4)"
  }));

  let width = 0;
  let height = 0;
  let dpr = 1;

  function resize() {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = rect.width;
    height = rect.height;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
  }

  resize();
  window.addEventListener("resize", resize, { passive: true });

  let angleY = 0;
  let angleX = 0.2;
  let targetAngleX = 0.2;
  let targetAngleY = 0;

  const parentCard = canvas.closest(".hero-card") || canvas;
  parentCard.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / rect.width - 0.5;
    const my = (e.clientY - rect.top) / rect.height - 0.5;
    targetAngleY = mx * 0.75;
    targetAngleX = -my * 0.5;
  }, { passive: true });

  parentCard.addEventListener("mouseleave", () => {
    targetAngleX = 0.2;
    targetAngleY = 0;
  }, { passive: true });

  function render(time) {
    if (!isVisible) return;
    animId = requestAnimationFrame(render);

    const elapsed = time - lastTime;
    if (elapsed < fpsInterval) return;
    lastTime = time - (elapsed % fpsInterval);

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    // Continuous smooth rotation + mouse interpolation
    angleY += 0.007;
    angleX += (targetAngleX - angleX) * 0.05;
    const curAngleY = angleY + targetAngleY;

    const cosY = Math.cos(curAngleY);
    const sinY = Math.sin(curAngleY);
    const cosX = Math.cos(angleX);
    const sinX = Math.sin(angleX);

    const centerX = width / 2;
    const centerY = height / 2 + 5;
    const fov = 380; // Perspective depth

    // Draw ambient floating probe particles
    ambientParticles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < -180) p.x = 180;
      if (p.x > 180) p.x = -180;
      if (p.y < -140) p.y = 140;
      if (p.y > 140) p.y = -140;

      const px = centerX + p.x;
      const py = centerY + p.y;
      ctx.beginPath();
      ctx.arc(px, py, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.fill();
    });

    // Project atoms into 2.5D coordinates
    const scaleFactor = Math.min(width, height) / 340;
    const projected = atoms.map(a => {
      const x1 = a.x * cosY - a.z * sinY;
      const z1 = a.z * cosY + a.x * sinY;
      const y2 = a.y * cosX - z1 * sinX;
      const z2 = z1 * cosX + a.y * sinX;

      const perspective = fov / (fov + z2 * 0.7);
      return {
        id: a.id,
        sym: a.sym,
        color: a.color,
        origR: a.r,
        x2d: centerX + x1 * scaleFactor * perspective,
        y2d: centerY + y2 * scaleFactor * perspective,
        z: z2,
        scale: perspective,
        r: Math.max(3, a.r * scaleFactor * perspective)
      };
    });

    // Draw bonds
    bonds.forEach(([u, v, order]) => {
      const a = projected[u];
      const b = projected[v];
      const avgZ = (a.z + b.z) / 2;
      const alpha = Math.max(0.2, Math.min(0.85, (avgZ + 120) / 240));

      ctx.save();
      ctx.strokeStyle = `rgba(148, 163, 184, ${alpha})`;
      ctx.lineWidth = Math.max(1.5, 2.2 * a.scale);
      ctx.lineCap = "round";

      if (order === 1) {
        ctx.beginPath();
        ctx.moveTo(a.x2d, a.y2d);
        ctx.lineTo(b.x2d, b.y2d);
        ctx.stroke();
      } else {
        const dx = b.x2d - a.x2d;
        const dy = b.y2d - a.y2d;
        const len = Math.hypot(dx, dy) || 1;
        const offX = (-dy / len) * 2.8 * a.scale;
        const offY = (dx / len) * 2.8 * a.scale;

        ctx.beginPath();
        ctx.moveTo(a.x2d + offX, a.y2d + offY);
        ctx.lineTo(b.x2d + offX, b.y2d + offY);
        ctx.moveTo(a.x2d - offX, a.y2d - offY);
        ctx.lineTo(b.x2d - offX, b.y2d - offY);
        ctx.stroke();
      }
      ctx.restore();
    });

    // Sort atoms by Z for painter's algorithm
    const sortedAtoms = [...projected].sort((a, b) => a.z - b.z);

    // Draw atoms with ultra-fast zero-allocation vector shading
    sortedAtoms.forEach(a => {
      ctx.save();
      // Main atom sphere
      ctx.beginPath();
      ctx.arc(a.x2d, a.y2d, a.r, 0, Math.PI * 2);
      ctx.fillStyle = a.color;
      ctx.fill();

      // Top-left 3D specular highlight
      ctx.beginPath();
      ctx.arc(a.x2d - a.r * 0.28, a.y2d - a.r * 0.28, a.r * 0.38, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 255, 255, 0.45)";
      ctx.fill();

      // Atom symbol text
      if (a.r > 5) {
        ctx.fillStyle = "#ffffff";
        ctx.font = `700 ${Math.max(8, Math.round(a.r * 1.02))}px Plus Jakarta Sans, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(a.sym, a.x2d, a.y2d + 0.5);
      }
      ctx.restore();
    });

    ctx.restore();
  }

  // Handle visibility & reduced motion
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    render(0);
    return;
  }

  if ("IntersectionObserver" in window) {
    const obs = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        isVisible = e.isIntersecting;
        if (isVisible && !animId) {
          lastTime = performance.now();
          animId = requestAnimationFrame(render);
        } else if (!isVisible && animId) {
          cancelAnimationFrame(animId);
          animId = null;
        }
      });
    }, { threshold: 0.05 });
    obs.observe(canvas);
  } else {
    animId = requestAnimationFrame(render);
  }
}

/* ─── Health check ────────────────────────────────────────── */
async function initHealthCheck() {
  const dot = $("statusDot");
  const text = $("statusText");
  try {
    const opts = {};
    if (typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function") {
      opts.signal = AbortSignal.timeout(4000);
    }
    const res = await fetch(`${API_BASE}/health`, opts);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    if (dot) dot.className = "status-dot online";
    if (text) text.textContent = "API online";
  } catch {
    if (dot) dot.className = "status-dot offline";
    if (text) text.textContent = "API offline";
  }
}

/* ─── Main 3D viewer (High-performance, stutter-free rotation) ─── */
function render3DMolecule(mol3dBlock, highlightIndices = null) {
  const container = $("mol3dContainer");
  const emptyEl = $("stageEmpty");
  const tools = $("viewerTools");
  const legend = $("atomLegend");
  if (!container) return;
  currentHighlight = highlightIndices;

  if (!mol3dBlock || !mol3dBlock.trim()) {
    if (viewer3D) viewer3D.clear();
    if (emptyEl) {
      emptyEl.hidden = false;
      const str = emptyEl.querySelector("strong");
      if (str) str.textContent = "3D model unavailable";
      const p = emptyEl.querySelector("p");
      if (p) p.textContent = "A 3D conformer could not be generated for this compound.";
    }
    if (tools) tools.hidden = true;
    if (legend) legend.hidden = true;
    return;
  }

  when3DmolReady((ok) => {
    if (!ok) {
      if (emptyEl) {
        emptyEl.hidden = false;
        const str = emptyEl.querySelector("strong");
        if (str) str.textContent = "3D engine failed to load";
        const p = emptyEl.querySelector("p");
        if (p) p.textContent = "Check your internet connection and refresh the page.";
      }
      return;
    }
    if (!viewer3D) {
      viewer3D = $3Dmol.createViewer(container, {
        backgroundColor: "white",
        backgroundAlpha: 1,
        antialias: false
      });

      // Pause spin immediately when dragging starts WITHOUT running heavy DOM scans
      container.addEventListener("mousedown", () => {
        if (isSpinning) {
          isSpinning = false;
          viewer3D.spin(false);
          updateSpinButtonUI(false);
        }
      });
      container.addEventListener("touchstart", () => {
        if (isSpinning) {
          isSpinning = false;
          viewer3D.spin(false);
          updateSpinButtonUI(false);
        }
      }, { passive: true });

      // Pause WebGL rendering when viewer is scrolled out of view
      if ("IntersectionObserver" in window) {
        const obs = new IntersectionObserver((entries) => {
          entries.forEach(e => {
            isViewer3DVisible = e.isIntersecting;
            if (viewer3D) {
              viewer3D.spin(isSpinning && isViewer3DVisible && !prefersReducedMotion ? "y" : false, 0.45);
            }
          });
        }, { threshold: 0.05 });
        obs.observe(container);
      }

      let resizeTimer;
      window.addEventListener("resize", () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => viewer3D && isViewer3DVisible && viewer3D.resize(), 150);
      }, { passive: true });
    }

    viewer3D.clear();
    viewer3D.addModel(mol3dBlock, "sdf");
    viewer3D.setStyle({}, styleFor(currentStyle));

    if (highlightIndices && highlightIndices.length) {
      const highlightSet = new Set(highlightIndices);
      const model = viewer3D.getModel();
      if (model && typeof model.getAtoms === "function") {
        const atoms = model.getAtoms();
        atoms.forEach(atom => {
          if (highlightSet.has(atom.index) && atom.style) {
            if (atom.style.sphere) delete atom.style.sphere.colorscheme;
            if (atom.style.stick) delete atom.style.stick.colorscheme;
          }
        });
      }
      const hl = currentStyle === "stick"
        ? { stick: { color: "#f59e0b", radius: 0.22 } }
        : { sphere: { color: "#f59e0b", scale: 0.38 }, stick: { color: "#f59e0b", radius: 0.18 } };
      viewer3D.setStyle({ index: highlightIndices }, hl);
    }

    viewer3D.zoomTo();
    viewer3D.zoom(0.72);
    viewer3D.render();
    if (emptyEl) emptyEl.hidden = true;
    if (tools) tools.hidden = false;
    if (legend) legend.hidden = false;
    setSpin(false);
  });
}

function updateSpinButtonUI(on) {
  const btn = $("btnToggleSpin");
  if (!btn) return;
  btn.setAttribute("aria-pressed", String(on));
  const span = btn.querySelector("span");
  if (span) span.textContent = on ? "Pause" : "Rotate";
  const icon = btn.querySelector("i, svg");
  if (icon) {
    icon.setAttribute("data-lucide", on ? "pause" : "play");
    if (window.lucide) window.lucide.createIcons({ root: btn });
  }
}

function setSpin(on) {
  isSpinning = on;
  if (viewer3D) {
    viewer3D.spin(on && isViewer3DVisible ? "y" : false, 0.45);
  }
  updateSpinButtonUI(on);
}

/* ─── Explainable AI ──────────────────────────────────────── */
function renderXAI(attribution) {
  const container = $("xaiContainer");
  const list = $("xaiSubstructuresList");
  const btnToggle = $("btnToggleAttribution");
  const btnText = $("btnAttributionText");
  if (!container || !list) return;

  const targetSpecs = $("targetSpecsCard");
  if (targetSpecs) targetSpecs.hidden = true;

  const subs = attribution?.top_substructures || [];
  if (!subs.length) { container.hidden = true; return; }

  container.hidden = false;
  list.innerHTML = "";
  isHighlightActive = false;
  if (btnText) btnText.textContent = "Highlight key atoms";
  const maxImp = Math.max(...subs.map(s => s.importance || 0), 1e-9);

  subs.forEach((sub, i) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "xai-chip";
    chip.style.animation = `rise .5s ${i * 70}ms var(--ease) both`;
    chip.innerHTML = `
      <div class="xai-chip-header">
        <span class="xai-chip-title">${escapeHtml(sub.description)}</span>
        <span class="xai-chip-weight">${(sub.importance * 100).toFixed(2)}%</span>
      </div>
      <div class="xai-bar"><span></span></div>
      <div class="xai-chip-body">
        <code>${escapeHtml(sub.fragment_smiles || "Fragment")}</code>
        <span class="xai-chip-count">${sub.atom_indices.length} atoms</span>
      </div>`;
    chip.addEventListener("click", () => {
      list.querySelectorAll(".xai-chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      render3DMolecule(currentMol3D, sub.atom_indices);
      if (btnText) btnText.textContent = "Reset colours";
      isHighlightActive = true;
      $("mol3dContainer")?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
    list.appendChild(chip);
    requestAnimationFrame(() => setTimeout(() => {
      chip.querySelector(".xai-bar span").style.width = `${(sub.importance / maxImp) * 100}%`;
    }, 80));
  });

  if (btnToggle) {
    btnToggle.onclick = () => {
      isHighlightActive = !isHighlightActive;
      if (isHighlightActive) {
        render3DMolecule(currentMol3D, attribution.highlighted_atom_indices);
        if (btnText) btnText.textContent = "Reset colours";
      } else {
        render3DMolecule(currentMol3D, null);
        list.querySelectorAll(".xai-chip").forEach(c => c.classList.remove("active"));
        if (btnText) btnText.textContent = "Highlight key atoms";
      }
    };
  }
}

/* ─── Lead optimisation & Active Application Suggestions ───────────── */
let cachedActiveSuggestions = [];
let showAllActiveApps = false;

function renderLeadOptimization(suggestions, prediction) {
  const container = $("optContainer");
  const list = $("optSuggestionsList");
  if (!container || !list) return;
  if (prediction !== "Inactive" || !suggestions?.length) { container.hidden = true; return; }

  container.hidden = false;
  list.innerHTML = suggestions.map((item, idx) => `
    <article class="opt-card" style="animation: rise .5s ${idx * 80}ms var(--ease) both">
      <div class="opt-card-header">
        <span class="opt-card-number">${idx + 1}</span>
        <h4 class="opt-card-title">${escapeHtml(item.title)}</h4>
        <span class="opt-card-impact">${escapeHtml(item.impact)}</span>
      </div>
      <p class="opt-card-rationale">${escapeHtml(item.rationale)}</p>
      <div class="opt-card-footer">
        <div><strong>Model driver:</strong> ${escapeHtml(item.target_feature)}</div>
        ${item.example_transformation ? `<div><strong>Example:</strong> ${escapeHtml(item.example_transformation)}</div>` : ""}
      </div>
    </article>`).join("");
}

function renderActiveApplications(suggestions, prediction) {
  const container = $("activeAppContainer");
  const list = $("activeAppList");
  if (!container || !list) return;
  if (prediction !== "Active" || !suggestions?.length) {
    container.hidden = true;
    return;
  }

  container.hidden = false;
  // Display strictly the top 3 most promising applications
  const displayList = suggestions.slice(0, 3);

  list.innerHTML = displayList.map((item, idx) => `
    <article class="opt-card app-card top-rec" style="animation: rise .5s ${idx * 80}ms var(--ease) both">
      <div class="opt-card-header">
        <span class="opt-card-number app-rank">#${idx + 1}</span>
        <h4 class="opt-card-title">${escapeHtml(item.title)}</h4>
        <span class="app-suitability-badge badge-top">
          <i data-lucide="star"></i> Top Match (${item.suitability_score}%)
        </span>
      </div>
      <div class="app-domain"><i data-lucide="cross"></i> ${escapeHtml(item.domain)}</div>
      <p class="opt-card-rationale">${escapeHtml(item.rationale)}</p>
      <div class="app-indications">
        <strong>Target Conditions:</strong> ${escapeHtml(item.target_conditions)}
      </div>
      <div class="opt-card-footer app-footer">
        <div><strong>RDKit Evidence:</strong> ${escapeHtml(item.rdkit_evidence)}</div>
        <div><strong>Reference Drugs:</strong> ${escapeHtml(item.reference_drugs)}</div>
      </div>
    </article>`).join("");

  if (window.lucide && typeof lucide.createIcons === "function") {
    lucide.createIcons();
  }
}

/* ─── Single prediction ───────────────────────────────────── */
function setVerdict(prediction, molName) {
  const banner = $("verdictBanner");
  const title = $("verdictTitle");
  const subtext = $("verdictSubtext");
  const icon = $("verdictIcon");
  const iconBox = $("verdictIconBox");
  const isActive = prediction === "Active";

  if (banner) {
    banner.className = `verdict-banner ${isActive ? "verdict-active" : "verdict-inactive"}`;
  }
  if (title) {
    title.textContent = isActive ? "BIOACTIVE (Active COX-2 Lead)" : "INACTIVE (Non-Inhibitory)";
  }
  if (subtext) {
    subtext.textContent = isActive
      ? "Predicted potent COX-2 inhibition (IC50 ≤ 1,000 nM). Molecule exhibits characteristic pharmacophore anchors."
      : "Predicted inactive or weak affinity (IC50 > 1,000 nM). Explore lead optimization suggestions below.";
  }
  if (icon) {
    icon.setAttribute("data-lucide", isActive ? "shield-check" : "alert-circle");
  }
  if (iconBox) {
    iconBox.className = `verdict-icon-box ${isActive ? "box-active" : "box-inactive"}`;
  }
  if (window.lucide && typeof lucide.createIcons === "function") {
    lucide.createIcons();
  }
}

function initSinglePrediction() {
  const input = $("smilesInput");
  const btnPredict = $("btnPredict");
  const errorBox = $("inputError");
  const empty = $("resultEmpty");
  const body = $("resultBody");
  const loading = $("resultLoading");
  const loadingStep = $("loadingStep");
  if (!input || !btnPredict) return;

  const showError = (msg) => {
    if (errorBox) {
      errorBox.textContent = msg;
      errorBox.classList.add("visible");
    }
  };
  const hideError = () => {
    if (errorBox) errorBox.classList.remove("visible");
  };

  $("btnExample")?.addEventListener("click", () => {
    input.value = EXAMPLES[exampleIdx++ % EXAMPLES.length];
    hideError();
    input.focus();
  });

  document.querySelectorAll(".qchip").forEach(chip => chip.addEventListener("click", () => {
    input.value = chip.dataset.value;
    hideError();
    btnPredict.click();
  }));

  $("btnClear")?.addEventListener("click", () => {
    input.value = "";
    hideError();
    if (empty) empty.hidden = false;
    if (body) body.hidden = true;
    if (loading) loading.hidden = true;
    if (viewer3D) { viewer3D.clear(); viewer3D.render(); }
    const stageEmpty = $("stageEmpty");
    if (stageEmpty) {
      stageEmpty.hidden = false;
      const str = stageEmpty.querySelector("strong");
      if (str) str.textContent = "No structure loaded";
      const p = stageEmpty.querySelector("p");
      if (p) p.textContent = "Run a prediction to render a force-field optimised 3D conformer.";
    }
    if ($("viewerTools")) $("viewerTools").hidden = true;
    if ($("atomLegend")) $("atomLegend").hidden = true;
    if ($("xaiContainer")) $("xaiContainer").hidden = true;
    if ($("optContainer")) $("optContainer").hidden = true;
    if ($("activeAppContainer")) $("activeAppContainer").hidden = true;
    if ($("targetSpecsCard")) $("targetSpecsCard").hidden = false;
  });

  input.addEventListener("keydown", (e) => {
    if (e.nativeEvent?.isComposing || e.isComposing || e.keyCode === 229) return;
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); btnPredict.click(); }
  });

  $("btnToggleSpin")?.addEventListener("click", () => setSpin(!isSpinning));
  $("btnResetView")?.addEventListener("click", () => {
    if (!viewer3D) return;
    viewer3D.zoomTo(); viewer3D.zoom(0.72); viewer3D.render();
  });
  $("btnSnapshot")?.addEventListener("click", () => {
    if (!viewer3D) return;
    const link = document.createElement("a");
    link.href = viewer3D.pngURI();
    link.download = `biopredict_${Date.now()}.png`;
    link.click();
    toast("Snapshot saved");
  });

  document.querySelectorAll("#styleSeg .seg-btn").forEach(btn => btn.addEventListener("click", () => {
    document.querySelectorAll("#styleSeg .seg-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentStyle = btn.dataset.style;
    if (currentMol3D) render3DMolecule(currentMol3D, currentHighlight);
  }));

  $("btnCopySmiles")?.addEventListener("click", async () => {
    const smilesText = $("resSmiles")?.textContent;
    if (smilesText && smilesText !== "—") {
      try { await navigator.clipboard.writeText(smilesText); toast("SMILES copied"); }
      catch { toast("Copy not available"); }
    }
  });

  const btnExport = $("btnExportDossier");
  btnExport?.addEventListener("click", async () => {
    const ctx = window.currentPredictionContext;
    if (!ctx || !ctx.smiles) {
      toast("Please run a prediction first.");
      return;
    }

    const origHtml = btnExport.innerHTML;
    btnExport.disabled = true;
    btnExport.innerHTML = '<i data-lucide="loader-circle" class="spin"></i><span>Generating PDF Dossier…</span>';
    if (window.lucide && typeof lucide.createIcons === "function") {
      lucide.createIcons();
    }

    try {
      const res = await fetch(`${API_BASE}/export_dossier`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          molecule_name: ctx.molecule_name,
          smiles: ctx.smiles,
          prediction: ctx.prediction,
          probability: ctx.probability,
          mol_info: ctx.mol_info,
          top_substructures: ctx.top_substructures,
          active_suggestions: ctx.active_suggestions,
          lead_optimization: ctx.lead_optimization,
          timestamp: new Date().toLocaleString()
        })
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const blob = await res.blob();
      const safeName = (ctx.molecule_name || "Compound").replace(/[^a-zA-Z0-9_\-]/g, "_");
      const filename = `Bioactivity_Dossier_${safeName}.pdf`;

      const blobUrl = window.URL.createObjectURL(blob);
      const dlLink = document.createElement("a");
      dlLink.href = blobUrl;
      dlLink.download = filename;
      document.body.appendChild(dlLink);
      dlLink.click();
      dlLink.remove();
      window.URL.revokeObjectURL(blobUrl);

      toast("Screening Dossier (PDF) downloaded!");
    } catch (err) {
      console.error("Export Dossier Error:", err);
      toast("Failed to export Dossier. Please try again.");
    } finally {
      btnExport.disabled = false;
      btnExport.innerHTML = origHtml;
      if (window.lucide && typeof lucide.createIcons === "function") {
        lucide.createIcons();
      }
    }
  });

  const steps = ["Resolving molecule…", "Generating Morgan fingerprint…", "Running XGBoost inference…", "Computing atom attribution…", "Building 3D conformer…"];

  btnPredict.addEventListener("click", async () => {
    const value = input.value.trim();
    if (!value) { showError("Please enter a molecule name or SMILES string."); input.focus(); return; }
    hideError();

    const stageEmpty = $("stageEmpty");
    if (stageEmpty) stageEmpty.hidden = true;
    if (loading) loading.hidden = false;
    btnPredict.disabled = true;
    setButtonContent(btnPredict, "loader-circle", "Analysing…");
    btnPredict.querySelector("svg")?.classList.add("spin");
    let s = 0;
    if (loadingStep) loadingStep.textContent = steps[0];
    const stepTimer = setInterval(() => {
      s = (s + 1) % steps.length;
      if (loadingStep) loadingStep.textContent = steps[s];
    }, 900);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles: value })
      });
      const data = await res.json();
      if (!res.ok || data.detail) throw new Error(data.detail || `Server error (${res.status})`);

      const isActive = data.prediction === "Active";
      setVerdict(data.prediction, data.molecule_name || value);

      const badge = $("predBadge");
      if (badge) {
        badge.textContent = data.prediction;
        badge.className = `badge ${isActive ? "badge-active" : "badge-inactive"}`;
      }

      if ($("resMolName")) $("resMolName").textContent = data.molecule_name || value;
      if ($("resModel")) $("resModel").textContent = data.model || "XGBoost";
      const info = data.mol_info || {};
      if ($("resFormula")) $("resFormula").textContent = info.formula || "—";
      if ($("resMW")) $("resMW").textContent = info.mol_weight ? `${info.mol_weight} g/mol` : "—";
      if ($("resAtoms")) $("resAtoms").textContent = info.num_atoms ?? "—";
      if ($("resBonds")) $("resBonds").textContent = info.num_bonds ?? "—";
      if ($("resSmiles")) $("resSmiles").textContent = data.smiles || "—";

      if (empty) empty.hidden = true;
      if (body) {
        body.hidden = false;
        body.style.animation = "none";
        void body.offsetWidth;
        body.style.animation = "rise .6s var(--ease) both";
      }

      currentMol3D = data.mol_3d || "";
      currentAttribution = data.atom_attribution || null;
      render3DMolecule(currentMol3D);
      renderXAI(currentAttribution);
      renderLeadOptimization(data.lead_optimization, data.prediction);
      renderActiveApplications(data.active_suggestions, data.prediction);

      // Update AI Assistant live context
      window.currentPredictionContext = {
        molecule_name: data.molecule_name || value,
        smiles: data.smiles || value,
        prediction: data.prediction,
        probability: data.probability,
        mol_info: data.mol_info || {},
        top_substructures: data.atom_attribution?.top_substructures || [],
        active_suggestions: data.active_suggestions || [],
        lead_optimization: data.lead_optimization || []
      };
      const contextBadge = $("chatMolContextBadge");
      if (contextBadge) {
        contextBadge.textContent = `${data.molecule_name || value} · ${data.prediction}`;
      }

      // Record in user history if authenticated
      const userToken = localStorage.getItem("biopredict_token");
      if (userToken) {
        fetch(`${API_BASE}/api/auth/history`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${userToken}`
          },
          body: JSON.stringify({
            molecule_name: data.molecule_name || value,
            smiles: data.smiles || value,
            prediction: data.prediction,
            probability: data.probability
          })
        }).catch(() => {});
      }
    } catch (err) {
      showError(err.message || "Failed to predict molecule.");
      if (!currentMol3D && $("stageEmpty")) $("stageEmpty").hidden = false;
    } finally {
      clearInterval(stepTimer);
      if (loading) loading.hidden = true;
      btnPredict.disabled = false;
      setButtonContent(btnPredict, "zap", "Predict bioactivity");
    }
  });
}

/* ─── Batch screening ─────────────────────────────────────── */
function initBatchPrediction() {
  const dropZone = $("dropZone");
  const fileInput = $("batchFileInput");
  const fileInfo = $("fileInfo");
  const fileName = $("fileName");
  const btnRun = $("btnRunBatch");
  const batchError = $("batchError");
  const results = $("batchResults");
  const tbody = $("batchTableBody");
  const progress = $("batchProgress");
  if (!dropZone || !fileInput) return;

  let selectedFile = null;
  let activeFilter = "all";

  const showError = (msg) => {
    if (batchError) {
      batchError.textContent = msg;
      batchError.classList.add("visible");
    }
  };

  $("btnBrowse").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  ["dragenter", "dragover"].forEach(ev => dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach(ev => dropZone.addEventListener(ev, () => dropZone.classList.remove("dragging")));
  dropZone.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]); });
  fileInput.addEventListener("change", (e) => { if (e.target.files.length) handleFile(e.target.files[0]); });

  $("btnRemoveFile").addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    fileInfo.hidden = true;
    btnRun.disabled = true;
  });

  const BATCH_ROW_LIMIT = 2000;

  function handleFile(file) {
    if (!file.name.toLowerCase().endsWith(".csv")) { showError("Please upload a valid .csv file."); return; }
    batchError.classList.remove("visible");

    // Count data rows client-side before allowing submission
    const reader = new FileReader();
    reader.onload = (e) => {
      const lines = e.target.result.split(/\r?\n/).filter(l => l.trim().length > 0);
      const dataRows = lines.length - 1; // subtract header row
      if (dataRows > BATCH_ROW_LIMIT) {
        showError(
          `Your CSV contains ${dataRows.toLocaleString()} rows — the maximum allowed is ${BATCH_ROW_LIMIT.toLocaleString()} rows. ` +
          `Please trim your file to ${BATCH_ROW_LIMIT.toLocaleString()} rows or fewer and re-upload.`
        );
        fileInfo.hidden = true;
        btnRun.disabled = true;
        return;
      }
      selectedFile = file;
      if (fileName) fileName.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB · ${dataRows.toLocaleString()} rows`;
      if (fileInfo) fileInfo.hidden = false;
      if (btnRun) btnRun.disabled = false;
    };
    reader.readAsText(file);
  }

  function renderRows() {
    if (!batchDataCache) return;
    const q = ($("batchSearch").value || "").trim().toLowerCase();
    const rows = batchDataCache.results.filter(r => {
      if (activeFilter === "error" && r.status === "ok") return false;
      if (activeFilter !== "all" && activeFilter !== "error" && r.prediction !== activeFilter) return false;
      return !q || String(r.smiles).toLowerCase().includes(q);
    });
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="4" class="muted-cell">No rows match this filter.</td></tr>`;
      return;
    }
    tbody.innerHTML = rows.slice(0, 500).map((r, i) => {
      const status = r.status === "ok"
        ? `<span class="st-ok">Valid</span>`
        : `<span class="st-err" title="${escapeHtml(r.error || "Invalid SMILES")}">Error</span>`;
      const smiles = String(r.smiles || "");
      const short = smiles.length > 38 ? escapeHtml(smiles.slice(0, 38)) + "…" : escapeHtml(smiles);
      const pred = r.prediction
        ? `<span class="badge badge-sm ${r.prediction === "Active" ? "badge-active" : "badge-inactive"}">${r.prediction}</span>` : "—";
      return `<tr class="row-in" style="animation-delay:${Math.min(i, 20) * 25}ms">
        <td>${r.row}</td><td><code title="${escapeHtml(smiles)}">${short}</code></td>
        <td>${status}</td><td>${pred}</td></tr>`;
    }).join("");
  }

  document.querySelectorAll("#batchFilters .seg-btn").forEach(btn => btn.addEventListener("click", () => {
    document.querySelectorAll("#batchFilters .seg-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    activeFilter = btn.dataset.filter;
    renderRows();
  }));
  $("batchSearch").addEventListener("input", renderRows);

  btnRun.addEventListener("click", async () => {
    if (!selectedFile) return;
    btnRun.disabled = true;
    setButtonContent(btnRun, "loader-circle", "Screening library…");
    btnRun.querySelector("svg")?.classList.add("spin");
    progress.hidden = false;
    batchError.classList.remove("visible");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch(`${API_BASE}/predict_batch`, { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Batch prediction failed.");
      batchDataCache = data;

      const active = data.results.filter(r => r.prediction === "Active").length;
      const inactive = data.results.filter(r => r.prediction === "Inactive").length;
      const valid = active + inactive;
      const total = data.total_rows || data.results.length;

      animateNumber($("statTotal"), total, 0, 900);
      animateNumber($("statActive"), active, 0, 900);
      animateNumber($("statInactive"), inactive, 0, 900);
      animateNumber($("statErrors"), data.invalid_rows || 0, 0, 900);
      animateNumber($("statHitRate"), valid ? (active / valid) * 100 : 0, 1, 900, "%");

      const donut = $("donut");
      if (donut && total) {
        donut.style.setProperty("--active", (active / total) * 100);
        donut.style.setProperty("--inactive", (inactive / total) * 100);
      }

      results.hidden = false;
      renderRows();
      if (data.truncated) {
        showError("⚠️ Your CSV had more than 2,000 rows. Only the first 2,000 were processed.");
      }
      toast(`Screened ${total} molecules`);
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      showError(err.message || "Failed to process batch screening.");
    } finally {
      progress.hidden = true;
      btnRun.disabled = !selectedFile;
      setButtonContent(btnRun, "play", "Start batch prediction");
    }
  });

  $("btnDownload").addEventListener("click", () => {
    if (!batchDataCache?.results) return;
    const rows = [
      ["Row", "SMILES", "Status", "Bioactivity_Verdict", "Error"],
      ...batchDataCache.results.map(r => [
        r.row, `"${r.smiles}"`, r.status, r.prediction || "", `"${r.error || ""}"`
      ])
    ];
    const blob = new Blob([rows.map(e => e.join(",")).join("\n")], { type: "text/csv" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `biopredict_results_${Date.now()}.csv`;
    link.click();
  });
}

/* ─── Benchmark ───────────────────────────────────────────── */
async function initBenchmark() {
  const bars = $("benchBars");
  const tbody = $("cvMetricsBody");
  if (!bars && !tbody) return;

  document.querySelectorAll("#splitSeg .seg-btn").forEach(btn => btn.addEventListener("click", () => {
    document.querySelectorAll("#splitSeg .seg-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    renderBenchmark(btn.dataset.split);
  }));

  try {
    const res = await fetch(`${API_BASE}/models`);
    modelsCache = await res.json();
    renderBenchmark("cv");
  } catch (err) {
    if (tbody) tbody.innerHTML = `<tr><td colspan="7" class="muted-cell">Metrics unavailable — is the API running?</td></tr>`;
    if (bars) bars.innerHTML = `<p class="muted-cell">Metrics unavailable.</p>`;
  }
}

function renderBenchmark(split) {
  if (!modelsCache) return;
  const list = (split === "holdout" ? modelsCache.holdout_results : modelsCache.cv_results) || [];
  if (!list.length) return;
  const sorted = [...list].sort((a, b) => (b.AUC || 0) - (a.AUC || 0));
  const pct = (v) => (v !== undefined && v !== null) ? (v * 100).toFixed(2) + "%" : "—";

  const bars = $("benchBars");
  if (bars) {
    bars.innerHTML = sorted.map(m => {
      const win = m.Model === "XGBoost";
      return `<div class="bar-row ${win ? "winner" : ""}">
        <span class="bar-name">${win ? '<i data-lucide="crown" class="crown"></i>' : ""}${escapeHtml(m.Model)}</span>
        <div class="bar-tracks">
          <div class="bar-track auc"><span data-w="${(m.AUC || 0) * 100}"></span></div>
          <div class="bar-track acc"><span data-w="${(m.ACC || 0) * 100}"></span></div>
        </div>
        <span class="bar-val">${((m.AUC || 0) * 100).toFixed(1)}</span>
      </div>`;
    }).join("");
    refreshIcons();
    requestAnimationFrame(() => setTimeout(() => {
      bars.querySelectorAll(".bar-track span").forEach(s => { s.style.width = s.dataset.w + "%"; });
    }, 60));
  }

  const cvBody = $("cvMetricsBody");
  if (cvBody) {
    cvBody.innerHTML = sorted.map(m => {
      const win = m.Model === "XGBoost";
      return `<tr class="${win ? "row-winner" : ""}">
        <td><strong>${escapeHtml(m.Model)}</strong>${win ? '<span class="winner-tag">Best</span>' : ""}</td>
        <td>${pct(m.ACC)}</td><td>${pct(m.Precision)}</td><td>${pct(m.SEN)}</td>
        <td>${pct(m.SPC)}</td><td>${pct(m.F1)}</td><td><strong>${pct(m.AUC)}</strong></td></tr>`;
    }).join("");
  }

  const best = list.find(m => m.Model === "XGBoost") || sorted[0];
  if ($("cmModel")) $("cmModel").textContent = best.Model;
  if ($("cmSplit")) $("cmSplit").textContent = split === "holdout" ? "hold-out test set" : "cross-validation";
  let tn = best.TN, fp = best.FP, fn = best.FN, tp = best.TP;
  if (tp === undefined && modelsCache.confusion_matrices) {
    const key = split === "holdout" ? Object.keys(modelsCache.confusion_matrices).find(k => k.includes("holdout")) : "validation_cv";
    const cm = modelsCache.confusion_matrices[key]?.[best.Model];
    if (cm) [[tn, fp], [fn, tp]] = cm;
  }
  [["cmTN", tn], ["cmFP", fp], ["cmFN", fn], ["cmTP", tp]].forEach(([id, v]) => {
    const el = $(id)?.querySelector("strong");
    if (el) { if (typeof v === "number") animateNumber(el, v, 0, 900); else el.textContent = "—"; }
  });
}

/* ─── Figure lightbox ─────────────────────────────────────── */
function initLightbox() {
  const dlg = $("lightbox");
  if (!dlg) return;
  document.querySelectorAll(".fig").forEach(fig => fig.addEventListener("click", () => {
    const img = $("lightboxImg");
    if (img) {
      img.src = fig.dataset.src;
      img.alt = fig.dataset.title;
    }
    const title = $("lightboxTitle");
    if (title) title.textContent = fig.dataset.title;
    dlg.showModal();
  }));
  $("lightboxClose")?.addEventListener("click", () => dlg.close());
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
}

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/* ─── BioPredict Molecular AI Assistant ───────────────────── */
function initChatbot() {
  const fab = $("chatFab");
  const drawer = $("chatDrawer");
  const btnClose = $("btnCloseChat");
  const btnClear = $("btnClearChat");
  const form = $("chatForm");
  const input = $("chatInput");
  const messagesBox = $("chatMessages");
  if (!fab || !drawer) return;

  const toggleDrawer = (open) => {
    const isOpening = (typeof open === "boolean") ? open : drawer.hidden;
    drawer.hidden = !isOpening;
    drawer.setAttribute("aria-hidden", String(!isOpening));
    if (isOpening) {
      if (input) input.focus();
      scrollChatBottom();
    }
  };

  fab.addEventListener("click", () => toggleDrawer());
  btnClose?.addEventListener("click", () => toggleDrawer(false));

  btnClear?.addEventListener("click", () => {
    if (!messagesBox) return;
    messagesBox.innerHTML = `
      <div class="chat-bubble bot">
        <p>👋 Conversation cleared. Ask me anything about the analyzed compound, COX-2 active site residues, or lead optimization!</p>
      </div>`;
  });

  const scrollChatBottom = () => {
    if (messagesBox) {
      messagesBox.scrollTop = messagesBox.scrollHeight;
    }
  };

  // Quick Chips
  document.querySelectorAll(".chat-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const q = chip.dataset.question;
      if (q && input) {
        input.value = q;
        form?.dispatchEvent(new Event("submit", { cancelable: true }));
      }
    });
  });

  function formatAiMarkdown(text) {
    if (!text) return "";
    let safe = escapeHtml(text);
    // Bold: **text**
    safe = safe.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    // Bullet lists
    const lines = safe.split("\n");
    let inList = false;
    const htmlLines = [];
    for (const l of lines) {
      const trimmed = l.trim();
      if (trimmed.startsWith("* ") || trimmed.startsWith("- ")) {
        if (!inList) { htmlLines.push("<ul>"); inList = true; }
        htmlLines.push(`<li>${trimmed.substring(2)}</li>`);
      } else {
        if (inList) { htmlLines.push("</ul>"); inList = false; }
        if (trimmed.length > 0) {
          htmlLines.push(`<p>${trimmed}</p>`);
        }
      }
    }
    if (inList) htmlLines.push("</ul>");
    return htmlLines.join("");
  }

  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = (input?.value || "").trim();
    if (!text) return;
    input.value = "";

    // Append user message
    const userBubble = document.createElement("div");
    userBubble.className = "chat-bubble user";
    userBubble.textContent = text;
    messagesBox.appendChild(userBubble);

    // Append loading placeholder
    const loadingBubble = document.createElement("div");
    loadingBubble.className = "chat-bubble bot loading-bubble";
    loadingBubble.innerHTML = `<span class="chat-dot"></span><span class="chat-dot"></span><span class="chat-dot"></span><span>Analyzing chemistry…</span>`;
    messagesBox.appendChild(loadingBubble);
    scrollChatBottom();

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          molecule_context: window.currentPredictionContext || null
        })
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Unable to reach molecular AI assistant.");
      }
      loadingBubble.className = "chat-bubble bot";
      loadingBubble.innerHTML = formatAiMarkdown(data.reply);
    } catch (err) {
      loadingBubble.className = "chat-bubble bot";
      loadingBubble.innerHTML = `<p style="color: var(--rose);">⚠️ ${escapeHtml(err.message)}</p>`;
    } finally {
      scrollChatBottom();
      refreshIcons();
    }
  });
}
