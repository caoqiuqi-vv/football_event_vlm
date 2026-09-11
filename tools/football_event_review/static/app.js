const $ = (selector) => document.querySelector(selector);
const COLORS = { shot: "#ff665c", save: "#55a8ff", set_piece: "#e0a6ff", free_kick: "#f2c75c", penalty: "#ff8d4f", corner: "#64d8cb", shot_on_target: "#87ef6c" };
const LABEL_NAMES = { shot: "射门", save: "扑救", set_piece: "定位球", free_kick: "任意球", penalty: "点球", corner: "角球", shot_on_target: "射正" };
const EVALUATION_NAMES = { fp: "模型 FP · 待人工核验", matched: "已匹配 GT", unlabeled: "该类别未标注", whistle_rescue: "哨声补漏 · 待人工确认" };
const STATUS_NAMES = { unreviewed: "未审核", accepted: "已保留", deleted: "已删除", modified: "已修改" };

const state = {
  bootstrap: null, video: null, currentVideoId: null, currentEventId: null,
  filtered: [], selectedLabel: null, correctedTime: null,
  viewStart: 0, viewEnd: 600, dragging: false, dragStartX: 0, dragViewStart: 0,
  hitboxes: [], contextEnd: null,
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try { message = (await response.json()).error || message; } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}

function formatTime(seconds, millis = true) {
  seconds = Math.max(0, Number(seconds) || 0);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  const base = hours ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}` : String(minutes).padStart(2, "0");
  return `${base}:${secs.toFixed(millis ? 3 : 0).padStart(millis ? 6 : 2, "0")}`;
}

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.classList.remove("show"), 1500);
}

function showSwitchFeedback(event) {
  const badge = $("#switchBadge");
  badge.textContent = `已切换 · ${LABEL_NAMES[event.label] || event.label} · ${formatTime(event.time_sec)}`;
  badge.classList.remove("hidden");
  clearTimeout(showSwitchFeedback.timer);
  showSwitchFeedback.timer = setTimeout(() => badge.classList.add("hidden"), 1100);
}

function currentEvent() {
  return state.video?.events.find((event) => event.id === state.currentEventId) || null;
}

function isConflict(event) {
  const predicted = event.label;
  const frame = event.frame_detection_scores || {};
  const frameTop = Object.entries(frame).sort((a, b) => b[1] - a[1])[0];
  return Number(frame[predicted] || 0) < 0.25 || (frameTop && frameTop[0] !== predicted && frameTop[1] > frame[predicted] + 0.12);
}

function applyFilters() {
  if (!state.video) return;
  const label = $("#labelFilter").value;
  const status = $("#statusFilter").value;
  const evaluation = $("#evaluationFilter").value;
  const conflictOnly = $("#conflictOnly").checked;
  const source = $("#sourceFilter")?.value || "all";
  state.filtered = state.video.events.filter((event) =>
    (label === "all" || event.label === label) &&
    (status === "all" || event.review.status === status) &&
    (evaluation === "all" || event.evaluation_status === evaluation) &&
    (!conflictOnly || isConflict(event)) &&
    (source === "all" ||
      (source === "model" && event.review_source !== "whistle_rescue") ||
      (source === "whistle_flagged" && event.review_source === "dino_frame_whistle") ||
      (source === "whistle_rescue" && event.review_source === "whistle_rescue"))
  );
  renderQueue();
  drawTimeline();
}

function renderProgress() {
  const counts = state.bootstrap.status_counts;
  const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
  const reviewed = total - counts.unreviewed;
  const percent = total ? reviewed / total * 100 : 0;
  $("#progressText").textContent = `${reviewed} / ${total} 已审核`;
  $("#progressPercent").textContent = `${percent.toFixed(1)}%`;
  $("#progressBar").style.width = `${percent}%`;
}

function renderQueue() {
  const list = $("#queueList");
  list.innerHTML = "";
  $("#queueCount").textContent = `${state.filtered.length} 条`;
  state.filtered.forEach((event) => {
    const button = document.createElement("button");
    button.className = `queue-item ${event.id === state.currentEventId ? "current" : ""} ${event.review.status !== "unreviewed" ? "reviewed" : ""} ${event.review.status}`;
    const whistle = Number(event.whistle_score || 0);
    button.innerHTML = `<i class="color" style="background:${COLORS[event.label]}"></i><time>${formatTime(event.time_sec)}</time><span class="label">${LABEL_NAMES[event.label]}${whistle ? ' <em>哨</em>' : ''}</span><span class="score">${event.review_source === 'whistle_rescue' ? `W ${whistle.toFixed(2)}` : event.score.toFixed(3)}</span>`;
    button.onclick = () => selectEvent(event.id, true);
    list.appendChild(button);
  });
  requestAnimationFrame(() => list.querySelector(".current")?.scrollIntoView({ block: "nearest" }));
}

function renderEvent() {
  const event = currentEvent();
  $("#emptyState").classList.toggle("hidden", Boolean(event));
  $("#eventCard").classList.toggle("hidden", !event);
  if (!event) return;
  const sourceLabel = event.label;
  const evaluationStatus = event.evaluation_status || "unlabeled";
  const frameScore = Number(event.frame_detection_scores?.[sourceLabel] || 0);
  const classElement = $("#eventClass");
  classElement.className = `event-class ${sourceLabel}`;
  classElement.textContent = sourceLabel.toUpperCase().replace("_", " ");
  $("#eventTime").textContent = formatTime(event.time_sec);
  const queueIndex = state.filtered.findIndex((item) => item.id === event.id);
  $("#eventPosition").textContent = `${queueIndex >= 0 ? queueIndex + 1 : "—"} / ${state.filtered.length}`;
  $("#evaluationBadge").className = `evaluation-badge ${evaluationStatus}`;
  $("#evaluationBadge").textContent = EVALUATION_NAMES[evaluationStatus] || evaluationStatus;
  $("#dinoScore").textContent = Number(event.score).toFixed(3);
  $("#frameScore").textContent = frameScore.toFixed(3);
  $("#dinoBar").style.width = `${Math.min(100, event.score * 100)}%`;
  $("#frameBar").style.width = `${Math.min(100, frameScore * 100)}%`;
  $("#supportRange").textContent = `片段 ${formatTime(event.support_start_sec)} – ${formatTime(event.support_end_sec)}`;
  $("#mergeCount").textContent = `合并 ${event.merged_predictions || 1} 个候选`;
  const whistleScore = Number(event.whistle_score || 0);
  $("#whistleEvidence").classList.toggle("hidden", whistleScore <= 0);
  if (whistleScore > 0) {
    $("#whistleScore").textContent = whistleScore.toFixed(3);
    $("#whistleSource").textContent = event.review_source === "whistle_rescue"
      ? "DINO 阈值外的独立补漏候选，请重点确认是否为定位球"
      : "当前模型候选附近检测到哨声，不增加额外观片";
  }
  const siblings = state.video.events.filter((item) => item.segment_id === event.segment_id && item.review.status === "unreviewed");
  const hasSetPiece = siblings.some((item) => item.label === "set_piece");
  $("#segmentBatchActions").classList.toggle("hidden", siblings.length < 2);
  $("#segmentBatchHint").textContent = `同一画面还有 ${siblings.length} 个未审核标签，可一次处理`;
  $("#acceptSegmentBtn").disabled = hasSetPiece;
  $("#acceptSegmentBtn").title = hasSetPiece ? "定位球必须先人工选择任意球/点球/角球等类型" : "一次保留该片段全部预测";

  $("#evidenceGrid").innerHTML = state.bootstrap.model_labels.map((label) =>
    `<div class="evidence-cell ${label === sourceLabel ? "active" : ""}"><span>${LABEL_NAMES[label]}</span><b>D ${Number(event.dino_scores?.[label] || 0).toFixed(2)} · F ${Number(event.frame_detection_scores?.[label] || 0).toFixed(2)}</b></div>`
  ).join("");

  state.selectedLabel = event.review.corrected_label || (state.bootstrap.review_labels.includes(sourceLabel) ? sourceLabel : null);
  state.correctedTime = event.review.corrected_time_sec ?? event.time_sec;
  $("#correctedTime").value = Number(state.correctedTime).toFixed(3);
  $("#note").value = event.review.note || "";
  renderClassButtons();
  const status = event.review.status;
  $("#statusPill").className = `status-pill ${status}`;
  $("#statusPill").textContent = STATUS_NAMES[status];
  $("#undoBtn").disabled = !event.review.revision;
}

function renderClassButtons() {
  $("#classButtons").innerHTML = state.bootstrap.review_labels.map((label, index) =>
    `<button data-label="${label}" class="${state.selectedLabel === label ? "selected" : ""}">${index + 1} ${LABEL_NAMES[label]}</button>`
  ).join("");
  $("#classButtons").querySelectorAll("button").forEach((button) => {
    button.onclick = () => { state.selectedLabel = button.dataset.label; renderClassButtons(); };
  });
}

function playEventContext(event) {
  const player = $("#player");
  if (event.review_source === "whistle_rescue") {
    player.currentTime = Math.max(0, Number(event.support_start_sec ?? event.start_sec));
    state.contextEnd = Math.min(state.video.duration_sec, Number(event.support_end_sec ?? event.end_sec));
  } else {
    player.currentTime = Math.max(0, event.time_sec - 3);
    state.contextEnd = Math.min(state.video.duration_sec, event.time_sec + 5);
  }
  const playback = player.play();
  if (playback) playback.catch(() => toast("浏览器阻止了自动播放，请点击画面或按 Space"));
}

function playFullSegment(event) {
  const player = $("#player");
  player.currentTime = Math.max(0, Number(event.support_start_sec ?? event.start_sec));
  state.contextEnd = Math.min(state.video.duration_sec, Number(event.support_end_sec ?? event.end_sec));
  player.play().catch(() => toast("浏览器阻止了自动播放，请点击画面或按 Space"));
}

async function saveSegmentDecision(status) {
  const event = currentEvent();
  if (!event) return;
  const siblings = state.video.events.filter((item) => item.segment_id === event.segment_id && item.review.status === "unreviewed");
  if (status === "accepted" && siblings.some((item) => item.label === "set_piece")) {
    toast("定位球需要先选择具体类型，请单独确认"); return;
  }
  const reviewer = $("#reviewerInput").value.trim();
  try {
    for (const item of siblings) {
      const updated = await api(`/api/events/${item.id}/decision`, { method: "POST", body: JSON.stringify({ status, reviewer, note: "同片段批量审核" }) });
      state.video.events[state.video.events.findIndex((candidate) => candidate.id === item.id)] = updated;
    }
    await refreshBootstrap(); applyFilters(); toast(`已一次处理 ${siblings.length} 个标签`); nextEvent(1, true, true);
  } catch (error) { toast(error.message); }
}

function selectEvent(eventId, seek = true, autoPlay = false) {
  state.currentEventId = eventId;
  const event = currentEvent();
  if (event) showSwitchFeedback(event);
  if (event && seek) {
    if (autoPlay) playEventContext(event);
    else $("#player").currentTime = Math.max(0, event.time_sec - 2.5);
    state.viewStart = Math.max(0, event.time_sec - 60);
    state.viewEnd = Math.min(state.video.duration_sec, state.viewStart + 120);
  }
  renderQueue(); renderEvent(); drawTimeline();
}

function nextEvent(direction = 1, onlyUnreviewed = false, autoPlay = true) {
  if (!state.video) return;
  const current = currentEvent();
  const ordered = state.video.events;
  const currentIndex = ordered.findIndex((event) => event.id === state.currentEventId);
  let candidates = state.filtered;
  if (onlyUnreviewed) candidates = ordered.filter((event) => event.review.status === "unreviewed");
  if (!candidates.length) { toast("当前筛选下没有待审核事件"); return; }
  let target = null;
  if (onlyUnreviewed && currentIndex >= 0) {
    target = direction > 0
      ? candidates.find((event) => ordered.indexOf(event) > currentIndex)
      : [...candidates].reverse().find((event) => ordered.indexOf(event) < currentIndex);
  } else {
    const candidateIndex = candidates.findIndex((event) => event.id === state.currentEventId);
    target = candidates[candidateIndex < 0 ? 0 : candidateIndex + direction];
  }
  if (!target) { toast(direction > 0 ? "已经是最后一个事件" : "已经是第一个事件"); return; }
  selectEvent(target.id, true, autoPlay);
}

function navigateAdjacent(direction) {
  if (!state.video) return;
  const current = currentEvent();
  const index = state.video.events.findIndex((event) => event.id === state.currentEventId);
  const target = state.video.events[index + direction];
  if (!target) { toast(direction > 0 ? "已经是最后一个事件" : "已经是第一个事件"); return; }
  const sameSegment = Boolean(current?.segment_id && current.segment_id === target.segment_id);
  selectEvent(target.id, !sameSegment, !sameSegment);
}

async function saveDecision(status, modified = false) {
  const event = currentEvent();
  if (!event) return;
  // Keep playback authorized by this click while the review request is saved.
  $("#player").play().catch(() => {});
  const payload = { status, reviewer: $("#reviewerInput").value.trim(), note: $("#note").value.trim() };
  if (modified) {
    payload.corrected_label = state.selectedLabel;
    payload.corrected_time_sec = Number($("#correctedTime").value);
  }
  try {
    const updated = await api(`/api/events/${event.id}/decision`, { method: "POST", body: JSON.stringify(payload) });
    const index = state.video.events.findIndex((item) => item.id === event.id);
    state.video.events[index] = updated;
    await refreshBootstrap();
    applyFilters();
    toast(status === "accepted" ? "已保留" : status === "deleted" ? "已删除" : "修改已保存");
    nextEvent(1, true, true);
  } catch (error) { toast(error.message); }
}

async function undoDecision() {
  const event = currentEvent();
  if (!event) return;
  try {
    const updated = await api(`/api/events/${event.id}/undo`, { method: "POST", body: "{}" });
    state.video.events[state.video.events.findIndex((item) => item.id === event.id)] = updated;
    await refreshBootstrap(); applyFilters(); renderEvent(); toast("已撤销");
  } catch (error) { toast(error.message); }
}

async function refreshBootstrap() {
  state.bootstrap = await api("/api/bootstrap");
  renderProgress();
  for (const option of $("#videoSelect").options) {
    const video = state.bootstrap.videos.find((item) => item.video_id === option.value);
    if (video) option.textContent = `${video.video_id}  ${video.reviewed}/${video.total}`;
  }
}

async function loadVideo(videoId) {
  state.currentVideoId = videoId;
  state.video = await api(`/api/videos/${videoId}`);
  $("#player").src = state.video.media_url;
  $("#videoBadge").textContent = videoId;
  state.viewStart = 0;
  state.viewEnd = Math.min(600, state.video.duration_sec || 600);
  applyFilters();
  const next = state.video.events.find((event) => event.review.status === "unreviewed") || state.video.events[0];
  if (next) selectEvent(next.id, true); else { state.currentEventId = null; renderEvent(); }
}

function setupTimeline() {
  const canvas = $("#timeline");
  const position = (event) => {
    const rect = canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top, rect };
  };
  canvas.addEventListener("click", (event) => {
    if (state.dragging) return;
    const { x, y, rect } = position(event);
    const hit = state.hitboxes.find((box) => Math.abs(box.x - x) < 8 && Math.abs(box.y - y) < 10);
    if (hit) return selectEvent(hit.eventId, true);
    const time = state.viewStart + (x - 62) / Math.max(1, rect.width - 76) * (state.viewEnd - state.viewStart);
    $("#player").currentTime = Math.max(0, Math.min(state.video.duration_sec, time));
  });
  canvas.addEventListener("wheel", (event) => {
    if (!state.video) return;
    event.preventDefault();
    const { x, rect } = position(event);
    const cursorRatio = Math.max(0, Math.min(1, (x - 62) / Math.max(1, rect.width - 76)));
    const oldSpan = state.viewEnd - state.viewStart;
    const newSpan = Math.max(30, Math.min(state.video.duration_sec, oldSpan * (event.deltaY > 0 ? 1.25 : 0.8)));
    const cursorTime = state.viewStart + cursorRatio * oldSpan;
    state.viewStart = Math.max(0, Math.min(state.video.duration_sec - newSpan, cursorTime - cursorRatio * newSpan));
    state.viewEnd = state.viewStart + newSpan;
    drawTimeline();
  }, { passive: false });
  canvas.addEventListener("mousedown", (event) => {
    state.dragging = false; state.dragStartX = event.clientX; state.dragViewStart = state.viewStart;
    const move = (moveEvent) => {
      if (Math.abs(moveEvent.clientX - state.dragStartX) > 4) state.dragging = true;
      if (!state.dragging) return;
      const span = state.viewEnd - state.viewStart;
      const delta = -(moveEvent.clientX - state.dragStartX) / canvas.clientWidth * span;
      state.viewStart = Math.max(0, Math.min(state.video.duration_sec - span, state.dragViewStart + delta));
      state.viewEnd = state.viewStart + span; drawTimeline();
    };
    const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); setTimeout(() => { state.dragging = false; }, 0); };
    window.addEventListener("mousemove", move); window.addEventListener("mouseup", up);
  });
  new ResizeObserver(drawTimeline).observe(canvas);
}

function drawTimeline() {
  const canvas = $("#timeline");
  if (!state.video || !canvas.clientWidth) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvas.clientWidth * ratio);
  canvas.height = Math.round(canvas.clientHeight * ratio);
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
  const width = canvas.clientWidth, height = canvas.clientHeight, left = 62, right = width - 14;
  const span = Math.max(1, state.viewEnd - state.viewStart);
  const xForTime = (time) => left + (time - state.viewStart) / span * (right - left);
  ctx.fillStyle = "#0c1117"; ctx.fillRect(0, 0, width, height);
  ctx.font = "10px ui-monospace, monospace"; ctx.textAlign = "center";
  const tickStep = span > 1800 ? 300 : span > 600 ? 120 : span > 240 ? 60 : span > 90 ? 30 : 10;
  const firstTick = Math.ceil(state.viewStart / tickStep) * tickStep;
  for (let time = firstTick; time <= state.viewEnd; time += tickStep) {
    const x = xForTime(time); ctx.strokeStyle = "#222b35"; ctx.beginPath(); ctx.moveTo(x, 20); ctx.lineTo(x, height); ctx.stroke();
    ctx.fillStyle = "#778392"; ctx.fillText(formatTime(time, false), x, 13);
  }
  state.hitboxes = [];
  const labels = state.bootstrap.model_labels;
  labels.forEach((label, rowIndex) => {
    const top = 28 + rowIndex * 57, bottom = top + 46;
    ctx.fillStyle = rowIndex % 2 ? "#10161d" : "#0e141a"; ctx.fillRect(left, top, right - left, bottom - top);
    ctx.fillStyle = COLORS[label]; ctx.textAlign = "right"; ctx.fillText(LABEL_NAMES[label], left - 9, top + 24);
    ctx.strokeStyle = "#252e39"; ctx.beginPath(); ctx.moveTo(left, bottom); ctx.lineTo(right, bottom); ctx.stroke();
    const points = state.video.timeline.filter((point) => point.end_sec >= state.viewStart && point.start_sec <= state.viewEnd);
    [["dino", "#f1f5fa", false], ["frame_detection", "#d7ff38", true]].forEach(([key, color, dashed]) => {
      ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = dashed ? 1.3 : 1.6; ctx.setLineDash(dashed ? [4, 3] : []);
      points.forEach((point, index) => {
        const x = xForTime((point.start_sec + point.end_sec) / 2);
        const y = bottom - 3 - Number(point[key]?.[label] || 0) * (bottom - top - 7);
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }); ctx.stroke(); ctx.setLineDash([]);
    });
    state.video.events.filter((event) => event.label === label && event.time_sec >= state.viewStart && event.time_sec <= state.viewEnd).forEach((event) => {
      const x = xForTime(event.time_sec), y = top + 7;
      const status = event.review.status;
      ctx.fillStyle = status === "deleted" ? "#55232b" : status === "accepted" ? "#5dd39e" : status === "modified" ? "#f1d470" : COLORS[label];
      ctx.beginPath(); ctx.moveTo(x, y - 5); ctx.lineTo(x + 5, y); ctx.lineTo(x, y + 5); ctx.lineTo(x - 5, y); ctx.closePath(); ctx.fill();
      if (event.id === state.currentEventId) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke(); }
      state.hitboxes.push({ x, y, eventId: event.id });
    });
  });
  const playerTime = $("#player").currentTime || 0;
  if (playerTime >= state.viewStart && playerTime <= state.viewEnd) {
    const x = xForTime(playerTime); ctx.strokeStyle = "#d7ff38"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, 18); ctx.lineTo(x, height); ctx.stroke();
    ctx.fillStyle = "#d7ff38"; ctx.beginPath(); ctx.moveTo(x - 4, 18); ctx.lineTo(x + 4, 18); ctx.lineTo(x, 24); ctx.fill();
  }
}

async function init() {
  state.bootstrap = await api("/api/bootstrap");
  renderProgress();
  $("#reviewerInput").value = localStorage.getItem("football-reviewer") || "";
  $("#reviewerInput").onchange = (event) => localStorage.setItem("football-reviewer", event.target.value);
  $("#videoSelect").innerHTML = state.bootstrap.videos.map((video) => `<option value="${video.video_id}">${video.video_id}  ${video.reviewed}/${video.total}</option>`).join("");
  $("#labelFilter").innerHTML += state.bootstrap.model_labels.map((label) => `<option value="${label}">${LABEL_NAMES[label]}</option>`).join("");
  setupTimeline();
  $("#videoSelect").onchange = (event) => loadVideo(event.target.value);
  ["#labelFilter", "#statusFilter", "#evaluationFilter", "#sourceFilter", "#conflictOnly"].forEach((selector) => $(selector).onchange = applyFilters);
  $("#nextUnreviewed").onclick = () => nextEvent(1, true, true);
  $("#prevEventBtn").onclick = () => navigateAdjacent(-1);
  $("#replayEventBtn").onclick = () => { const event = currentEvent(); if (event) playEventContext(event); };
  $("#playFullSegment").onclick = () => { const event = currentEvent(); if (event) playFullSegment(event); };
  $("#acceptSegmentBtn").onclick = () => saveSegmentDecision("accepted");
  $("#deleteSegmentBtn").onclick = () => saveSegmentDecision("deleted");
  $("#nextEventBtn").onclick = () => navigateAdjacent(1);
  $("#acceptBtn").onclick = () => saveDecision("accepted");
  $("#deleteBtn").onclick = () => saveDecision("deleted");
  $("#saveModify").onclick = () => saveDecision("modified", true);
  $("#undoBtn").onclick = undoDecision;
  document.querySelectorAll("[data-seek]").forEach((button) => button.onclick = () => { $("#player").currentTime = Math.max(0, $("#player").currentTime + Number(button.dataset.seek)); });
  document.querySelectorAll("[data-adjust]").forEach((button) => button.onclick = () => { const input = $("#correctedTime"); input.value = Math.max(0, Number(input.value) + Number(button.dataset.adjust)).toFixed(3); });
  $("#usePlayerTime").onclick = () => { $("#correctedTime").value = $("#player").currentTime.toFixed(3); };
  $("#playContext").onclick = () => { const event = currentEvent(); if (!event) return; const player = $("#player"); player.currentTime = Math.max(0, event.time_sec - 5); state.contextEnd = event.time_sec + 8; player.play(); };
  const player = $("#player");
  $("#soundBtn").onclick = () => {
    player.muted = false; player.volume = 1;
    player.play().catch(() => {});
    $("#soundBtn").textContent = "🔊 声音已开启";
    setTimeout(() => $("#soundBtn").classList.add("hidden"), 1200);
  };
  const showBuffering = () => $("#bufferBadge").classList.remove("hidden");
  const hideBuffering = () => $("#bufferBadge").classList.add("hidden");
  player.onwaiting = showBuffering;
  player.onstalled = showBuffering;
  player.onseeking = showBuffering;
  player.oncanplay = hideBuffering;
  player.onplaying = hideBuffering;
  player.onseeked = hideBuffering;
  player.ontimeupdate = () => { $("#timeBadge").textContent = formatTime(player.currentTime); if (state.contextEnd && player.currentTime >= state.contextEnd) { player.pause(); state.contextEnd = null; } drawTimeline(); };
  window.addEventListener("keydown", (event) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    if (event.code === "Space") { event.preventDefault(); player.paused ? player.play() : player.pause(); }
    else if (event.key === "Enter") saveDecision("accepted");
    else if (["x", "X", "Delete", "Backspace"].includes(event.key)) saveDecision("deleted");
    else if (["j", "J", "ArrowLeft"].includes(event.key)) navigateAdjacent(-1);
    else if (["k", "K", "ArrowRight"].includes(event.key)) navigateAdjacent(1);
    else if (["r", "R"].includes(event.key)) { const current = currentEvent(); if (current) playEventContext(current); }
    else if (["1", "2", "3", "4", "5", "6"].includes(event.key)) { state.selectedLabel = state.bootstrap.review_labels[Number(event.key) - 1]; renderClassButtons(); }
  });
  if (state.bootstrap.videos.length) await loadVideo(state.bootstrap.videos[0].video_id);
}

init().catch((error) => { console.error(error); toast(`载入失败：${error.message}`); });
