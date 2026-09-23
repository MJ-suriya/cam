/**
 * Dahua Multi-Camera AI Face & Fingerprint Dual-Verification Application Logic
 * Powered by Local Tailwind CSS Dark Theme
 */

let allAttendanceRecords = [];
let allVerificationRecords = [];
let activeWindows = [];
let allEmployees = [];
let activeChannels = [];
let isFocused = false;

document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initTabs();
  loadCamerasGrid();
  initWebSocket();
  loadVerificationData();
  loadAttendanceData();
  loadEmployeesData();

  // Polling for active 5-minute countdowns
  setInterval(loadActiveWindows, 2000);

  // Grid Controls
  document.getElementById("btn3Grid")?.addEventListener("click", () => resetCameraFocus());
  document.getElementById("btnReloadCams")?.addEventListener("click", () => reloadCameraStreams());
  document.getElementById("attendanceSearch")?.addEventListener("input", (e) => filterAttendanceTable(e.target.value));
  document.getElementById("verificationSearch")?.addEventListener("input", (e) => filterVerificationTable(e.target.value));
});

// Live Clock
function initClock() {
  const clockEl = document.getElementById("liveClock");
  const dateEl = document.getElementById("liveDate");

  function update() {
    const now = new Date();
    clockEl.textContent = now.toLocaleTimeString("en-US", { hour12: false });
    dateEl.textContent = now.toLocaleDateString("en-US", {
      weekday: "short",
      day: "2-digit",
      month: "short",
      year: "numeric"
    });
  }

  update();
  setInterval(update, 1000);
}

// Tab Switching with Explicit Display Management
function switchTab(targetTab) {
  const tabs = ["surveillance", "verification", "attendance", "employees"];

  // Update button active styling
  const navBtns = document.querySelectorAll(".nav-btn");
  navBtns.forEach((b) => {
    const tabAttr = b.getAttribute("data-tab");
    if (tabAttr === targetTab) {
      b.classList.add("active", "bg-sky-600", "text-white");
      b.classList.remove("text-slate-400");
    } else {
      b.classList.remove("active", "bg-sky-600", "text-white");
      b.classList.add("text-slate-400");
    }
  });

  // Explicitly manage display property for each tab pane
  tabs.forEach((t) => {
    const pane = document.getElementById(`tab-${t}`);
    if (pane) {
      if (t === targetTab) {
        pane.style.setProperty("display", "flex", "important");
        pane.classList.remove("hidden");
        pane.classList.add("active");
      } else {
        pane.style.setProperty("display", "none", "important");
        pane.classList.add("hidden");
        pane.classList.remove("active");
      }
    }
  });

  // Reload data for the chosen tab
  if (targetTab === "verification") {
    loadVerificationData();
  } else if (targetTab === "attendance") {
    loadAttendanceData();
  } else if (targetTab === "employees") {
    loadEmployeesData();
  }
}

function initTabs() {
  const navBtns = document.querySelectorAll(".nav-btn");
  navBtns.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const targetTab = btn.getAttribute("data-tab");
      if (targetTab) {
        switchTab(targetTab);
      }
    });
  });

  // Initialize initial tab view explicitly
  switchTab("surveillance");
}

// Dynamic Camera Grid Loader from config/channels_config.py
async function loadCamerasGrid() {
  try {
    const res = await fetch("/api/channels");
    activeChannels = await res.json();

    const grid = document.getElementById("camsGrid");
    const badgeText = document.getElementById("activeChannelsBadge");
    if (!grid) return;

    if (badgeText) {
      const channelIds = activeChannels.map((c) => c.id).join(", ");
      badgeText.textContent = `NVR 2 • Channels ${channelIds}`;
    }

    grid.innerHTML = "";
    activeChannels.forEach((cam) => {
      const card = document.createElement("div");
      card.id = `camCard${cam.id}`;
      card.className = "cam-card bg-[#0D121F] border border-slate-800 rounded-xl overflow-hidden flex flex-col shadow-lg";
      card.innerHTML = `
        <div class="bg-[#0A0E18] px-3 py-2 border-b border-slate-800 flex items-center justify-between">
          <div class="flex items-center gap-2 text-xs font-semibold text-white">
            <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span class="font-mono text-sky-400 font-bold">CH ${cam.id}</span>
            <span class="text-slate-300 truncate max-w-[150px]">${cam.name}</span>
          </div>
          <button onclick="focusCamera(${cam.id})" class="text-slate-400 hover:text-white text-xs px-2 py-1 rounded bg-slate-800/60" title="Maximize View">⛶</button>
        </div>
        <div class="flex-1 bg-black relative flex items-center justify-center min-h-[220px]">
          <img src="/video_feed/${cam.id}" class="w-full h-full object-contain" id="stream${cam.id}" alt="CH ${cam.id}">
          <div class="absolute bottom-2 right-2 bg-slate-900/90 border border-emerald-500/30 px-2 py-0.5 rounded text-[10px] font-bold text-emerald-400 backdrop-blur-sm">
            AI Active &bull; &ge;87%
          </div>
        </div>
      `;
      grid.appendChild(card);
    });
  } catch (e) {
    console.error("[Cameras] Error loading channels:", e);
  }
}

// Camera Focus Mode
function focusCamera(ch) {
  const grid = document.getElementById("camsGrid");
  const cards = document.querySelectorAll(".cam-card");

  cards.forEach((card) => card.classList.add("hidden"));

  const targetCard = document.getElementById(`camCard${ch}`);
  if (targetCard) {
    targetCard.classList.remove("hidden");
    grid.className = "flex-1 grid grid-cols-1 gap-3 min-h-0";
    isFocused = true;
  }
}

function resetCameraFocus() {
  const grid = document.getElementById("camsGrid");
  const cards = document.querySelectorAll(".cam-card");
  cards.forEach((card) => card.classList.remove("hidden"));
  grid.className = "flex-1 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 min-h-0 overflow-y-auto";
  isFocused = false;
}

function reloadCameraStreams() {
  if (!activeChannels || activeChannels.length === 0) {
    loadCamerasGrid();
    return;
  }
  activeChannels.forEach((cam) => {
    const img = document.getElementById(`stream${cam.id}`);
    if (img) {
      const src = img.src.split("?")[0];
      img.src = `${src}?t=${Date.now()}`;
    }
  });
}

// WebSocket Real-Time Attendance Stream
function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/attendance`;

  const ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("[WebSocket] Connected to live verification stream");
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "new_attendance" || data.type === "attendance_event") {
        const record = data.record || data.data || data;
        if (record && (record.ecno || record.name)) {
          addLiveAttendanceCard(record);
          loadAttendanceData();
          loadVerificationData();
        }
      } else if (data.type === "verification_update" || data.type === "verification_deleted") {
        loadVerificationData();
        loadActiveWindows();
      }
    } catch (e) {
      console.error("[WebSocket] Parse error:", e);
    }
  };

  ws.onclose = () => {
    console.log("[WebSocket] Disconnected. Reconnecting in 3s...");
    setTimeout(initWebSocket, 3000);
  };
}

function addLiveAttendanceCard(record) {
  const container = document.getElementById("liveEventsContainer");
  const emptyState = document.getElementById("emptyStateEvents");
  if (emptyState) {
    emptyState.style.display = "none";
  }

  const rawEcno = record.ecno || record.emp_id || "";
  const isEmployee = (rawEcno && rawEcno !== "UNREGISTERED" && rawEcno !== "undefined") || record.status === "PRESENT";
  const empName = record.name || record.emp_name || (isEmployee ? `Employee ${rawEcno}` : "Visitor / Person");
  const empIdText = isEmployee ? `ID: ${rawEcno}` : "Visitor / Unregistered";
  const badgeText = isEmployee ? (record.final_status || "VERIFIED") : "VISITOR";
  const badgeBg = isEmployee ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" : "bg-amber-500/20 text-amber-400 border-amber-500/30";
  const borderLeft = isEmployee ? "border-l-4 border-l-emerald-500" : "border-l-4 border-l-amber-500";

  const card = document.createElement("div");
  card.className = `flex items-center gap-3 bg-gradient-to-r from-slate-900 to-[#121927] border border-slate-800 ${borderLeft} rounded-xl p-3 shadow-md hover:border-slate-700 transition-all`;

  const defaultSvg = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='48' height='48' viewBox='0 0 24 24' fill='%2364748b'><path d='M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z'/></svg>";
  const avatarUrl = record.snapshot ? `/saved_faces/${record.snapshot}` : (isEmployee ? `/profiledb/${rawEcno}.jpeg` : defaultSvg);
  const timeStr = record.check_in_time || record.morning_face_time || record.time || new Date().toLocaleTimeString();
  const camSrc = record.camera_source || record.camera || "NVR 2";

  card.innerHTML = `
    <img src="${avatarUrl}" class="w-12 h-12 rounded-xl object-cover border-2 border-slate-700 shadow-md shrink-0 cursor-pointer hover:scale-105 hover:border-sky-400 transition-all" alt="${empName}" onclick="showPersonPhotoModal('${avatarUrl}', '${empName.replace(/'/g, "\\'")}', '${rawEcno}', '${record.confidence ? Math.round(record.confidence * 100) + '%' : '95%'}', '${timeStr}', '${badgeText}')" onerror="this.onerror=null; this.src='${defaultSvg}';">
    <div class="flex-1 min-w-0 cursor-pointer" onclick="showPersonPhotoModal('${avatarUrl}', '${empName.replace(/'/g, "\\'")}', '${rawEcno}', '${record.confidence ? Math.round(record.confidence * 100) + '%' : '95%'}', '${timeStr}', '${badgeText}')">
      <div class="text-xs font-bold text-white truncate">${empName}</div>
      <div class="text-[11px] font-mono text-sky-400">${empIdText}</div>
      <div class="flex items-center gap-2 text-[10px] text-slate-400 mt-0.5">
        <span>⏰ ${timeStr}</span>
        <span>•</span>
        <span>${camSrc}</span>
      </div>
    </div>
    <div class="px-2 py-0.5 rounded-full text-[10px] font-bold border uppercase ${badgeBg}">${badgeText}</div>
  `;

  container.insertBefore(card, container.firstChild);

  while (container.children.length > 25) {
    container.removeChild(container.lastChild);
  }
}

// ==========================================
// DUAL-VERIFICATION MATRIX DATA & RENDERING
// ==========================================
async function loadVerificationData() {
  try {
    const res = await fetch("/api/verification/today");
    const data = await res.json();
    allVerificationRecords = data.records || [];
    activeWindows = data.active_windows || [];

    // Metrics calculation
    let validCount = 0;
    let pendingCount = activeWindows.length;
    let missingCount = 0;
    let exitCount = 0;

    allVerificationRecords.forEach((r) => {
      if (r.morning_status === "VALID" || r.evening_status === "VALID") validCount++;
      if (r.morning_status === "FINGERPRINT_MISSING" || r.evening_status === "FINGERPRINT_MISSING") missingCount++;
      if (r.exit_time) exitCount++;
    });

    document.getElementById("vMetricValid").textContent = validCount;
    document.getElementById("vMetricPending").textContent = pendingCount;
    document.getElementById("vMetricMissing").textContent = missingCount;
    document.getElementById("vMetricExits").textContent = exitCount;

    document.getElementById("todayCountBadge").textContent = `${validCount} Verified`;

    renderVerificationTable(allVerificationRecords);
    renderActiveWindows(activeWindows);
  } catch (e) {
    console.error("[Verification] Error loading data:", e);
  }
}

function renderVerificationTable(records) {
  const tbody = document.getElementById("verificationTableBody");
  if (!tbody) return;
  tbody.innerHTML = "";

  if (records.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" class="text-center py-12 text-slate-500 font-medium">No verification events recorded today.</td></tr>`;
    return;
  }

  const defaultSvg = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='36' height='36' viewBox='0 0 24 24' fill='%2364748b'><path d='M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z'/></svg>";

  records.forEach((r) => {
    const tr = document.createElement("tr");
    tr.className = "hover:bg-slate-800/40 transition-all border-b border-slate-800/40";

    const avatarUrl = r.snapshot ? `/saved_faces/${r.snapshot}` : (r.ecno && r.ecno !== "UNREGISTERED" && r.ecno !== "undefined" ? `/profiledb/${r.ecno}.jpeg` : defaultSvg);
    const morningPill = getStatusPill(r.morning_status);
    const eveningPill = getStatusPill(r.evening_status);
    const finalPill = getFinalStatusPill(r);
    const confScore = r.confidence ? `${Math.round(r.confidence * 100)}%` : "95%";
    const timeDisplay = r.morning_face_time || r.evening_face_time || r.morning_fp_time || "Today";

    tr.innerHTML = `
      <td class="p-3.5">
        <div class="flex items-center gap-3">
          <div class="relative group cursor-pointer" onclick="showPersonPhotoModal('${avatarUrl}', '${r.name.replace(/'/g, "\\'")}', '${r.ecno}', '${confScore}', '${timeDisplay}', '${r.final_status || r.morning_status}')">
            <img src="${avatarUrl}" class="w-11 h-11 rounded-xl object-cover border-2 border-slate-700 shadow group-hover:border-sky-400 group-hover:scale-105 transition-all shrink-0 bg-black" alt="${r.name}" onerror="this.onerror=null; this.src='${defaultSvg}';">
            <span class="absolute -bottom-1 -right-1 bg-sky-500 text-[9px] text-white font-bold rounded px-1 shadow opacity-0 group-hover:opacity-100 transition-opacity">🔍</span>
          </div>
          <div class="cursor-pointer" onclick="showPersonPhotoModal('${avatarUrl}', '${r.name.replace(/'/g, "\\'")}', '${r.ecno}', '${confScore}', '${timeDisplay}', '${r.final_status || r.morning_status}')">
            <div class="font-bold text-white leading-tight hover:text-sky-300 transition-colors">${r.name}</div>
            <div class="font-mono text-[11px] text-sky-400 font-semibold">ID: ${r.ecno}</div>
          </div>
        </div>
      </td>
      <td class="p-3.5 font-mono text-xs">${r.morning_face_time ? `<span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-200">📷 ${r.morning_face_time}</span>` : `<span class="text-slate-500">—</span>`}</td>
      <td class="p-3.5 font-mono text-xs">${r.morning_fp_time ? `<span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-200">👆 ${r.morning_fp_time}</span>` : `<span class="text-slate-500">—</span>`}</td>
      <td class="p-3.5">${morningPill}</td>
      <td class="p-3.5">${r.exit_remark ? `<span class="text-xs text-purple-400 font-medium bg-purple-500/10 px-2.5 py-1 rounded border border-purple-500/20">🚪 ${r.exit_remark}</span>` : `<span class="text-slate-500">—</span>`}</td>
      <td class="p-3.5 font-mono text-xs">${r.evening_face_time ? `<span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-200">📷 ${r.evening_face_time}</span>` : `<span class="text-slate-500">—</span>`}</td>
      <td class="p-3.5 font-mono text-xs">${r.evening_fp_time ? `<span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-200">👆 ${r.evening_fp_time}</span>` : `<span class="text-slate-500">—</span>`}</td>
      <td class="p-3.5">${finalPill}</td>
      <td class="p-3.5 text-center">
        <div class="flex items-center justify-center gap-1.5">
          <button onclick="openEditRecordModal('${r.date}', '${r.ecno}')" class="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-sky-400 border border-slate-700 text-xs font-semibold" title="Alter Record">✏️ Edit</button>
          <button onclick="deleteVerificationRecord('${r.date}', '${r.ecno}')" class="px-2 py-1 rounded bg-slate-800 hover:bg-red-900/60 text-red-400 border border-slate-700 text-xs font-semibold" title="Delete Record">🗑️ Delete</button>
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function getStatusPill(status) {
  if (status === "VALID") {
    return `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 uppercase">✓ VALID</span>`;
  } else if (status === "PENDING_FINGERPRINT") {
    return `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-bold bg-amber-500/15 text-amber-400 border border-amber-500/30 uppercase animate-pulse">⏳ WAITING FP</span>`;
  } else if (status === "FINGERPRINT_MISSING") {
    return `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-bold bg-red-500/15 text-red-400 border border-red-500/30 uppercase">⚠️ FP MISSING</span>`;
  }
  return `<span class="text-slate-500">—</span>`;
}

function getFinalStatusPill(record) {
  if (record.morning_status === "VALID" || record.evening_status === "VALID") {
    return `<span class="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 uppercase">✓ VALID ATTENDANCE</span>`;
  } else if (record.morning_status === "PENDING_FINGERPRINT" || record.evening_status === "PENDING_FINGERPRINT") {
    return `<span class="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-bold bg-amber-500/20 text-amber-400 border border-amber-500/40 uppercase animate-pulse">⏳ 5M WINDOW OPEN</span>`;
  } else if (record.morning_status === "FINGERPRINT_MISSING" || record.evening_status === "FINGERPRINT_MISSING") {
    return `<span class="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-bold bg-red-500/20 text-red-400 border border-red-500/40 uppercase">❌ FP NOT VERIFIED</span>`;
  }
  return `<span class="px-2 py-0.5 rounded text-[11px] bg-slate-800 text-slate-400">PENDING</span>`;
}

function filterVerificationTable(query) {
  const q = query.toLowerCase().trim();
  if (!q) {
    renderVerificationTable(allVerificationRecords);
    return;
  }
  const filtered = allVerificationRecords.filter((r) => {
    return (
      (r.name && r.name.toLowerCase().includes(q)) ||
      (r.ecno && r.ecno.toLowerCase().includes(q)) ||
      (r.morning_status && r.morning_status.toLowerCase().includes(q)) ||
      (r.evening_status && r.evening_status.toLowerCase().includes(q)) ||
      (r.exit_remark && r.exit_remark.toLowerCase().includes(q))
    );
  });
  renderVerificationTable(filtered);
}

// Active 5-Minute Window Ticker
async function loadActiveWindows() {
  try {
    const res = await fetch("/api/verification/active_windows");
    const data = await res.json();
    renderActiveWindows(data.active_windows || []);
  } catch (e) {}
}

function renderActiveWindows(windows) {
  const box = document.getElementById("activeWindowsBox");
  const list = document.getElementById("activeWindowsList");
  if (!box || !list) return;

  if (windows.length === 0) {
    box.classList.add("hidden");
    list.innerHTML = "";
    return;
  }

  box.classList.remove("hidden");
  list.innerHTML = "";

  windows.forEach((w) => {
    const mins = Math.floor(w.remaining_seconds / 60);
    const secs = w.remaining_seconds % 60;
    const timeFormatted = `${mins}:${secs < 10 ? "0" : ""}${secs}`;

    const item = document.createElement("div");
    item.className = "flex items-center justify-between bg-slate-900/90 border border-amber-500/30 rounded-lg px-2.5 py-1.5 text-xs";
    item.innerHTML = `
      <span class="font-bold text-white truncate">${w.name} (${w.ecno})</span>
      <span class="font-mono font-bold text-amber-400 shrink-0">⏳ ${timeFormatted}</span>
    `;
    list.appendChild(item);
  });
}

// ==========================================
// ALTER / EDIT & DELETE RECORD ACTIONS
// ==========================================
function openEditRecordModal(dateStr, ecno) {
  const record = allVerificationRecords.find((r) => r.date === dateStr && r.ecno === ecno);
  if (!record) return;

  document.getElementById("editDate").value = dateStr;
  document.getElementById("editEcno").value = ecno;
  document.getElementById("editEmpName").textContent = record.name;
  document.getElementById("editEmpId").textContent = `ID: ${record.ecno}`;

  document.getElementById("editMorningFace").value = record.morning_face_time || "";
  document.getElementById("editMorningFp").value = record.morning_fp_time || "";
  document.getElementById("editMorningStatus").value = record.morning_status || "NOT_CHECKED_IN";
  document.getElementById("editExitRemark").value = record.exit_remark || "";

  document.getElementById("editEveningFace").value = record.evening_face_time || "";
  document.getElementById("editEveningFp").value = record.evening_fp_time || "";
  document.getElementById("editEveningStatus").value = record.evening_status || "NOT_CHECKED_IN";

  document.getElementById("editRecordModal")?.classList.remove("hidden");
}

function closeEditRecordModal() {
  document.getElementById("editRecordModal")?.classList.add("hidden");
}

async function handleEditRecordSubmit(event) {
  event.preventDefault();
  const dateStr = document.getElementById("editDate").value;
  const ecno = document.getElementById("editEcno").value;

  const payload = {
    morning_face_time: document.getElementById("editMorningFace").value.trim() || null,
    morning_fp_time: document.getElementById("editMorningFp").value.trim() || null,
    morning_status: document.getElementById("editMorningStatus").value,
    exit_remark: document.getElementById("editExitRemark").value.trim() || null,
    evening_face_time: document.getElementById("editEveningFace").value.trim() || null,
    evening_fp_time: document.getElementById("editEveningFp").value.trim() || null,
    evening_status: document.getElementById("editEveningStatus").value
  };

  try {
    const res = await fetch(`/api/verification/record/${dateStr}/${ecno}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    if (result.success) {
      alert("✓ Attendance verification record altered successfully!");
      closeEditRecordModal();
      loadVerificationData();
    } else {
      alert("Update failed: " + result.error);
    }
  } catch (e) {
    alert("Error altering record: " + e.message);
  }
}

async function deleteVerificationRecord(dateStr, ecno) {
  if (!confirm(`Are you sure you want to delete the verification record for Employee ${ecno} on ${dateStr}?`)) {
    return;
  }

  try {
    const res = await fetch(`/api/verification/record/${dateStr}/${ecno}`, {
      method: "DELETE"
    });
    const result = await res.json();
    if (result.success) {
      alert("✓ Record deleted successfully.");
      loadVerificationData();
    } else {
      alert("Delete failed: " + result.error);
    }
  } catch (e) {
    alert("Error deleting record: " + e.message);
  }
}

async function deleteAttendanceRecord(dateStr, ecno) {
  if (!confirm(`Are you sure you want to delete the attendance log record for Employee ${ecno}?`)) {
    return;
  }

  try {
    const res = await fetch(`/api/attendance/record/${dateStr}/${ecno}`, {
      method: "DELETE"
    });
    const result = await res.json();
    if (result.success) {
      alert("✓ Attendance log record deleted successfully.");
      loadAttendanceData();
    }
  } catch (e) {
    alert("Error deleting record: " + e.message);
  }
}

// ==========================================
// ATTENDANCE LOG (LEGACY COMPATIBILITY)
// ==========================================
async function loadAttendanceData() {
  try {
    const res = await fetch("/api/attendance/today");
    const data = await res.json();
    allAttendanceRecords = data.records || [];
    renderAttendanceTable(allAttendanceRecords);
  } catch (e) {
    console.error("[Attendance] Error loading data:", e);
  }
}

function renderAttendanceTable(records) {
  const tbody = document.getElementById("attendanceTableBody");
  if (!tbody) return;
  tbody.innerHTML = "";

  if (records.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" class="text-center py-12 text-slate-500 font-medium">No attendance logged today.</td></tr>`;
    return;
  }

  const defaultSvg = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='36' height='36' viewBox='0 0 24 24' fill='%2364748b'><path d='M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z'/></svg>";

  records.forEach((r) => {
    const tr = document.createElement("tr");
    tr.className = "hover:bg-slate-800/40 transition-all border-b border-slate-800/40";

    const avatarUrl = r.snapshot ? `/saved_faces/${r.snapshot}` : (r.ecno && r.ecno !== "UNREGISTERED" && r.ecno !== "undefined" ? `/profiledb/${r.ecno}.jpeg` : defaultSvg);
    const confPct = Math.round((r.confidence || 0.95) * 100);

    tr.innerHTML = `
      <td class="p-3.5">
        <div class="relative group inline-block cursor-pointer" onclick="showPersonPhotoModal('${avatarUrl}', '${r.name.replace(/'/g, "\\'")}', '${r.ecno}', '${confPct}%', '${r.check_in_time}', '${r.status}')">
          <img src="${avatarUrl}" class="w-11 h-11 rounded-xl object-cover border-2 border-slate-700 shadow group-hover:border-sky-400 group-hover:scale-105 transition-all bg-black" alt="${r.name}" onerror="this.onerror=null; this.src='${defaultSvg}';">
        </div>
      </td>
      <td class="p-3.5 font-mono text-sky-400 font-bold">${r.ecno}</td>
      <td class="p-3.5 font-bold text-white">${r.name}</td>
      <td class="p-3.5 text-slate-300 font-mono text-xs">${r.date}</td>
      <td class="p-3.5 text-slate-300 font-mono text-xs">⏰ ${r.check_in_time}</td>
      <td class="p-3.5 text-slate-300 text-xs">${r.camera_source}</td>
      <td class="p-3.5"><span class="px-2 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20 font-mono font-bold text-xs">${confPct}%</span></td>
      <td class="p-3.5"><span class="px-2.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 text-xs font-bold uppercase">${r.status}</span></td>
      <td class="p-3.5 text-center">
        <button onclick="deleteAttendanceRecord('${r.date}', '${r.ecno}')" class="px-2 py-1 rounded bg-slate-800 hover:bg-red-900/60 text-red-400 border border-slate-700 text-xs font-semibold" title="Delete Log Record">🗑️ Delete</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function filterAttendanceTable(query) {
  const q = query.toLowerCase().trim();
  if (!q) {
    renderAttendanceTable(allAttendanceRecords);
    return;
  }
  const filtered = allAttendanceRecords.filter((r) => {
    return (
      (r.name && r.name.toLowerCase().includes(q)) ||
      (r.ecno && r.ecno.toLowerCase().includes(q)) ||
      (r.camera_source && r.camera_source.toLowerCase().includes(q))
    );
  });
  renderAttendanceTable(filtered);
}

// ==========================================
// EMPLOYEE ROSTER
// ==========================================
async function loadEmployeesData() {
  try {
    const res = await fetch("/api/employees");
    allEmployees = await res.json();

    const punchSelect = document.getElementById("punchEcnoSelect");
    if (punchSelect) {
      punchSelect.innerHTML = "";
      allEmployees.forEach((emp) => {
        const opt = document.createElement("option");
        opt.value = emp.ecno;
        opt.textContent = `${emp.name} (ID: ${emp.ecno})`;
        punchSelect.appendChild(opt);
      });
    }

    const grid = document.getElementById("employeesGrid");
    if (!grid) return;
    grid.innerHTML = "";

    const defaultSvg = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='80' height='80' viewBox='0 0 24 24' fill='%2364748b'><path d='M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z'/></svg>";

    allEmployees.forEach((emp) => {
      const card = document.createElement("div");
      card.className = "bg-[#0D121F] border border-slate-800 rounded-xl p-4 flex flex-col items-center text-center shadow-lg hover:border-sky-500/50 hover:shadow-sky-950/30 transition-all cursor-pointer group";
      card.onclick = () => showPersonPhotoModal(`/profiledb/${emp.image}`, emp.name, emp.ecno, '100% Enrolled', 'Enrolled Database', 'ROSTER PROFILE');
      card.innerHTML = `
        <img src="/profiledb/${emp.image}" class="w-20 h-20 rounded-2xl object-cover border-2 border-slate-700 group-hover:border-sky-400 group-hover:scale-105 transition-all mb-3 bg-black shadow" alt="${emp.name}" onerror="this.onerror=null; this.src='${defaultSvg}';">
        <div class="font-bold text-white text-xs truncate w-full group-hover:text-sky-300 transition-colors">${emp.name}</div>
        <div class="font-mono text-sky-400 font-semibold text-[11px] mt-0.5">ID: ${emp.ecno}</div>
      `;
      grid.appendChild(card);
    });
  } catch (e) {
    console.error("[Employees] Error loading data:", e);
  }
}

// ==========================================
// MODALS MANAGEMENT
// ==========================================
function openEnrollModal() {
  document.getElementById("enrollModal")?.classList.remove("hidden");
}

function closeEnrollModal() {
  document.getElementById("enrollModal")?.classList.add("hidden");
}

async function handleEnrollSubmit(event) {
  event.preventDefault();
  const submitBtn = document.getElementById("enrollSubmitBtn");
  submitBtn.disabled = true;
  submitBtn.textContent = "Saving...";

  const formData = new FormData();
  formData.append("ecno", document.getElementById("enrollEcno").value.trim());
  formData.append("name", document.getElementById("enrollName").value.trim());
  formData.append("photo", document.getElementById("enrollPhoto").files[0]);

  try {
    const res = await fetch("/api/employees/enroll", {
      method: "POST",
      body: formData
    });
    const result = await res.json();
    if (result.success) {
      alert(`✓ Successfully enrolled ${result.message}`);
      closeEnrollModal();
      document.getElementById("enrollForm").reset();
      loadEmployeesData();
    } else {
      alert(`Enrollment failed: ${result.message}`);
    }
  } catch (e) {
    alert(`Error: ${e.message}`);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Save & Enroll";
  }
}

// Settings Modal
async function openSettingsModal() {
  try {
    const res = await fetch("/api/config/verification");
    const cfg = await res.json();
    document.getElementById("settingFaceThresh").value = Math.round(cfg.face_confidence_threshold * 100);
    document.getElementById("settingWindowMins").value = cfg.fingerprint_window_minutes;
    document.getElementById("settingMorningStart").value = (cfg.morning_start_time || "09:00:00").substring(0, 5);
    document.getElementById("settingEveningStart").value = (cfg.evening_start_time || "19:30:00").substring(0, 5);
    document.getElementById("settingCooldown").value = cfg.cooldown_seconds || 60;
  } catch (e) {
    console.error("[Settings] Error loading config:", e);
  }
  document.getElementById("settingsModal")?.classList.remove("hidden");
}

function closeSettingsModal() {
  document.getElementById("settingsModal")?.classList.add("hidden");
}

async function handleSettingsSubmit(event) {
  event.preventDefault();
  const faceThreshPct = parseFloat(document.getElementById("settingFaceThresh").value);
  const windowMins = parseInt(document.getElementById("settingWindowMins").value);
  const morningStart = document.getElementById("settingMorningStart").value + ":00";
  const eveningStart = document.getElementById("settingEveningStart").value + ":00";
  const cooldownSec = parseInt(document.getElementById("settingCooldown").value);

  const payload = {
    face_confidence_threshold: faceThreshPct / 100.0,
    fingerprint_window_minutes: windowMins,
    morning_start_time: morningStart,
    evening_start_time: eveningStart,
    cooldown_seconds: cooldownSec
  };

  try {
    const res = await fetch("/api/config/verification", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      alert("✓ Verification settings updated successfully!");
      closeSettingsModal();
      loadVerificationData();
    }
  } catch (e) {
    alert("Error updating settings: " + e.message);
  }
}

// Simulate Fingerprint Punch Modal
function openSimulatePunchModal() {
  document.getElementById("punchModal")?.classList.remove("hidden");
}

function closeSimulatePunchModal() {
  document.getElementById("punchModal")?.classList.add("hidden");
}

async function handlePunchSubmit(event) {
  event.preventDefault();
  const ecno = document.getElementById("punchEcnoSelect").value;
  const deviceId = document.getElementById("punchDeviceId").value;

  try {
    const res = await fetch("/api/fingerprint/simulate_punch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ecno, device_id: deviceId })
    });
    const result = await res.json();
    if (result.success) {
      alert(`✓ Biometric Fingerprint punch recorded for ID: ${ecno}!\nReconciliation will automatically confirm valid attendance.`);
      closeSimulatePunchModal();
      loadVerificationData();
    }
  } catch (e) {
    alert("Error recording punch: " + e.message);
  }
}

function exportVerificationCSV() {
  window.location.href = "/api/verification/export";
}

function exportAttendanceCSV() {
  window.location.href = "/api/attendance/export";
}

// ==========================================
// PHOTO PREVIEW MODAL
// ==========================================
function showPersonPhotoModal(imgUrl, name, ecno, confidence, timeStr, status) {
  const modal = document.getElementById("photoPreviewModal");
  if (!modal) return;

  const img = document.getElementById("photoModalImg");
  const nameEl = document.getElementById("photoModalName");
  const ecnoEl = document.getElementById("photoModalEcno");
  const badgeEl = document.getElementById("photoModalConfidenceBadge");
  const statusEl = document.getElementById("photoModalStatus");
  const timeEl = document.getElementById("photoModalTime");

  if (img) img.src = imgUrl;
  if (nameEl) nameEl.textContent = name || "Employee";
  if (ecnoEl) ecnoEl.textContent = ecno ? `Employee ID: ${ecno}` : "Unregistered";
  if (badgeEl) badgeEl.textContent = confidence ? `${confidence} Match` : "Verified Match";
  if (statusEl) statusEl.textContent = status || "VALID ATTENDANCE";
  if (timeEl) timeEl.textContent = timeStr ? `⏰ Captured / Verified: ${timeStr}` : `⏰ Verified Today`;

  modal.classList.remove("hidden");
}

function closePhotoPreviewModal() {
  const modal = document.getElementById("photoPreviewModal");
  if (modal) modal.classList.add("hidden");
}

// Escape key closes modal
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closePhotoPreviewModal();
    closeEditRecordModal();
    closeEnrollModal();
    closeSettingsModal();
    closeSimulatePunchModal();
  }
});

