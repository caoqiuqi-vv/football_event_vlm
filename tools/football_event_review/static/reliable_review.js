/* v40 functions override legacy declarations after all HTML/JS patches. */
let reviewOutbox = null;
let videoLoadGeneration = 0;
let videoLoadController = null;
let videoLoading = false;
let outboxReady = false;

async function api(path, options = {}) {
  const controller = new AbortController();
  const upstream = options.signal;
  const cancel = () => controller.abort();
  if (upstream?.aborted) cancel();
  upstream?.addEventListener('abort', cancel, {once: true});
  const timeout = setTimeout(cancel, 10000);
  try {
    const response = await fetch(path, {...options, signal: controller.signal,
      headers: {'Content-Type': 'application/json', ...(options.headers || {})}});
    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      try { message = (await response.json()).error || message; } catch (_) {}
      const error = new Error(message); error.status = response.status; throw error;
    }
    return await response.json();
  } finally { clearTimeout(timeout); upstream?.removeEventListener('abort', cancel); }
}

function newOperationId() {
  return [...crypto.getRandomValues(new Uint8Array(16))].map(x => x.toString(16).padStart(2, '0')).join('');
}

function currentSaveEvent() {
  if (!outboxReady || videoLoading || !state.video || state.video.video_id !== state.currentVideoId) {
    toast('正在加载视频或恢复保存队列，请稍候'); return null;
  }
  return currentEvent();
}

function queueOperation(kind, event, payload) {
  const videoId = state.currentVideoId;
  const target = kind === 'team' ? videoId : event.id;
  const key = kind === 'team' ? `${videoId}:team` : `${videoId}:segment:${segmentKey(event)}`;
  const id = newOperationId();
  const path = kind === 'team' ? `/api/videos/${target}/team-profile` : `/api/events/${target}/${kind === 'undo' ? 'undo' : 'segment-decision'}`;
  return reviewOutbox.add({id, key, kind, target, videoId, path, payload: {...payload, operation_id: id}});
}

function markLocalPending(item) {
  if (!state.video || state.currentVideoId !== item.videoId || item.kind !== 'segment') return;
  const selected = new Set(item.payload.selected_labels || []);
  const expected = item.payload.expected_revisions || {};
  state.video.events.forEach(event => {
    if (!Object.hasOwn(expected, event.id)) return;
    event.review = {...event.review, pending_local: true,
      status: item.payload.review_outcome === 'needs_confirmation' ? 'needs_confirmation' : selected.has(event.label) ? 'accepted' : 'deleted'};
  });
}

function overlayPending() {
  if (!reviewOutbox) return;
  for (const item of reviewOutbox.list()) {
    if (['pending', 'retry', 'auth'].includes(item.status)) markLocalPending(item);
  }
}

function renderSaveQueue() {
  const status = document.querySelector('#saveQueueStatus');
  if (!status || !reviewOutbox) return;
  const items = reviewOutbox.list();
  const issues = items.filter(x => ['conflict', 'failed', 'editing', 'auth'].includes(x.status));
  status.textContent = items.length ? `待保存 ${items.length} 条${issues.length ? ` · 待处理 ${issues.length} 条` : ''}${reviewOutbox.running ? ' · 保存中' : ''}` : '全部操作已保存';
  document.querySelector('#saveQueuePanel').dataset.failed = String(issues.length > 0);
  const container = document.querySelector('#saveQueueItems'); container.replaceChildren();
  for (const item of items) {
    const row = document.createElement('div');
    const text = document.createElement('p');
    text.textContent = `${item.videoId} · ${item.target}：${item.error || '等待保存'}${item.status === 'editing' ? '（草稿保留中）' : ''}`;
    row.append(text);
    if (['conflict', 'failed', 'editing'].includes(item.status)) {
      const edit = document.createElement('button'); edit.textContent = '读取最新结果并重新编辑';
      edit.onclick = () => editQueuedOperation(item).catch(error => toast(error.message)); row.append(edit);
      const discard = document.createElement('button'); discard.textContent = '放弃这条草稿';
      discard.onclick = () => { if (window.confirm('仅删除本浏览器中的这条失败草稿，不改变已保存标注。确定放弃？')) reviewOutbox.discard(item.id); }; row.append(discard);
    } else {
      const retry = document.createElement('button'); retry.textContent = item.status === 'auth' ? '登录后重试' : '立即重试';
      retry.onclick = () => reviewOutbox.retry(item.id); row.append(retry);
      if (item.status === 'auth') {
        const login = document.createElement('a'); login.href = '/'; login.textContent = '打开登录页'; login.target = '_blank'; row.append(login);
      }
    }
    container.append(row);
  }
}

async function editQueuedOperation(item) {
  const generation = await loadVideo(item.videoId);
  if (!generation || state.currentVideoId !== item.videoId) return;
  if (item.kind === 'segment' || item.kind === 'undo') {
    if (!state.video.events.some(event => event.id === item.target)) throw new Error('事件已变化，原操作仍保留在队列中');
    selectEvent(item.target, true, false);
    const p = item.payload;
    if (item.kind === 'segment') {
      state.selectedSegmentLabels = new Set(p.selected_labels || []);
      state.secondaryLabels = new Set(Object.values(p.secondary_labels_by_label || {}).flat());
      state.attributionByLabel = structuredClone(p.attribution_by_label || {});
      if (p.corrected_time_sec !== undefined) $('#correctedTime').value = p.corrected_time_sec;
      $('#note').value = p.note || '';
      renderSegmentLabelEditor(); renderTeamAttribution();
    }
  } else {
    state.teamSetupOpen = true; renderTeamSetup();
    const p = item.payload;
    $('#teamAColor').value = p.teams.teamA.hex; $('#teamAName').value = p.teams.teamA.display_name;
    $('#teamBColor').value = p.teams.teamB.hex; $('#teamBName').value = p.teams.teamB.display_name;
    $('#periodSplitSec').value = p.period_split_sec;
    $('#firstPeriodLeftTeam').value = p.first_period_left_team; $('#secondPeriodLeftTeam').value = p.second_period_left_team;
  }
  reviewOutbox.edit(item.id);
  toast('已读取最新版本并恢复本地草稿，请核对后重新确认；原操作继续保留');
}

function setupReliableQueue() {
  if (reviewOutbox) return;
  const userId = state.bootstrap?.current_user?.user_id;
  const namespace = state.bootstrap?.review_namespace;
  if (!userId || !namespace) throw new Error('服务版本已变化，请刷新页面');
  reviewOutbox = new ReviewOutbox({storage: localStorage, scope: `${namespace}:${userId}`,
    send: item => api(item.path, {method: 'POST', body: JSON.stringify(item.payload)}),
    onChange: renderSaveQueue,
    onSaved: async (item, result) => {
      if (state.currentVideoId === item.videoId && state.video && !videoLoading) {
        if (item.kind === 'segment') mergeSavedSegment(result);
        else if (item.kind === 'team') {
          if (result.revision >= state.video.team_profile.revision) {
            state.video.team_profile = result; state.video.team_palette = result.teams;
            renderTeamSetup(); renderTeamAttribution();
          }
        } else {
          const event = state.video.events.find(event => event.id === result.id);
          if (event && result.review.revision >= event.review.revision) Object.assign(event, result);
        }
        overlayPending(); applyFilters();
      }
      scheduleBootstrapRefresh();
    }});
  // Fail closed if storage is unavailable/full; no optimistic success before durability.
  const probe = `football-review-v40-probe:${newOperationId()}`;
  localStorage.setItem(probe, '1'); localStorage.removeItem(probe);
  reviewOutbox.list(); outboxReady = true;
  $('#saveQueueToggle').onclick = () => { const panel = $('#saveQueueDetails'); panel.hidden = !panel.hidden; };
  window.addEventListener('beforeunload', event => {
    if (reviewOutbox.list().length) { event.preventDefault(); event.returnValue = ''; }
  });
  window.addEventListener('online', () => reviewOutbox.resume().catch(error => toast(error.message)));
  window.addEventListener('storage', event => {
    if (event.key?.startsWith(reviewOutbox.prefix)) { renderSaveQueue(); reviewOutbox.drain().catch(console.error); }
  });
  const player = $('#player'), retryVideo = $('#retryVideo');
  if (typeof installForwardBuffer === 'function') installForwardBuffer(player, () => {
    const current = state.video?.events.find(event => event.id === state.currentEventId);
    if (!current || videoLoading) return null;
    const next = state.video.events.find(event =>
      event.time_sec > current.time_sec && segmentKey(event) !== segmentKey(current) &&
      event.review?.status === 'unreviewed');
    return next ? Number(next.support_start_sec ?? next.start_sec ?? next.time_sec) : null;
  });
  player.addEventListener('error', () => { retryVideo.hidden = false; toast('视频加载失败，可点击“重新加载视频”重试'); });
  player.addEventListener('loadeddata', () => { retryVideo.hidden = true; });
  let bufferNoticeTimer = null;
  const buffering = () => {
    clearTimeout(bufferNoticeTimer);
    bufferNoticeTimer = setTimeout(() => {
      if (player.error || !player.getAttribute('src') || (!player.seeking && player.readyState >= 3)) return;
      retryVideo.title = player.readyState === 0 ? '正在读取视频索引，请稍候；长时间无画面可重试。' : '视频缓冲中，长时间无画面可重试。';
      retryVideo.textContent = '视频缓冲慢，重试';
      retryVideo.hidden = false;
    }, 10000);
  };
  const buffered = () => {
    if (player.seeking || player.readyState < 2) return;
    clearTimeout(bufferNoticeTimer); retryVideo.title = ''; retryVideo.textContent = '重新加载视频'; retryVideo.hidden = true;
  };
  for (const event of ['loadstart', 'waiting', 'seeking', 'stalled']) player.addEventListener(event, buffering);
  for (const event of ['canplay', 'playing', 'seeked']) player.addEventListener(event, buffered);
  player.addEventListener('emptied', () => { clearTimeout(bufferNoticeTimer); retryVideo.title = ''; retryVideo.textContent = '重新加载视频'; });
  player.addEventListener('error', () => { clearTimeout(bufferNoticeTimer); retryVideo.title = '视频连接或解码失败，请重试。'; });

  retryVideo.onclick = async () => {
    const videoId = state.currentVideoId, eventId = state.currentEventId;
    const target = state.pendingAbsoluteSeek?.target ?? player.currentTime;
    if (await loadVideo(videoId)) {
      if (state.video.events.some(event => event.id === eventId)) selectEvent(eventId, false);
      seekAbsolute(target, true);
    }
  };
  renderSaveQueue();
  reviewOutbox.resume().catch(error => toast(`保存队列恢复失败：${error.message}`));
}

function mergeSavedSegment(result) {
  if (!state.video || result.video_id !== state.currentVideoId || !result.segment_events) return;
  const incoming = new Map(result.segment_events.map(event => [event.id, event]));
  // A delayed idempotent receipt must not overwrite a newer loaded revision.
  for (const event of state.video.events) {
    const saved = incoming.get(event.id);
    if (saved && event.review.revision > saved.review.revision) incoming.set(event.id, event);
  }
  const untouched = state.video.events.filter(event => !incoming.has(event.id));
  state.video.events = [...untouched, ...incoming.values()].sort((a, b) => a.time_sec - b.time_sec || a.label.localeCompare(b.label));
}

async function saveMultiLabelSegment() {
  const event = currentSaveEvent(); if (!event) return;
  const selectedLabels = [...state.selectedSegmentLabels];
  if (!guardGtDeletion(new Set(selectedLabels))) return;
  if (selectedLabels.includes('set_piece') && !(state.bootstrap.set_piece_type_labels || []).some(label => state.secondaryLabels.has(label))) {
    toast('请先选择具体定位球类型'); return;
  }
  let attribution;
  try { attribution = confirmedAttribution(selectedLabels); }
  catch (error) { toast(error.message); return; }
  const time = Number($('#correctedTime').value);
  if (!Number.isFinite(time) || time < 0 || time > state.video.duration_sec) { toast('事件时间超出视频范围'); return; }
  const payload = {selected_labels: selectedLabels,
    secondary_labels_by_label: {
      shot: [...state.secondaryLabels].filter(label => (state.bootstrap.shot_detail_labels || []).includes(label)),
      save: [], set_piece: [...state.secondaryLabels].filter(label => (state.bootstrap.set_piece_type_labels || []).includes(label))},
    active_label: event.label, corrected_time_sec: time, note: $('#note').value.trim(),
    attribution_by_label: attribution, apply_time_to_all: true,
    compact_response: true,
    expected_revisions: Object.fromEntries(segmentEvents(event).map(item => [item.id, item.review.revision || 0]))};
  try {
    const item = queueOperation('segment', event, payload);
    markLocalPending(item); applyFilters(); toast('已加入保存队列');
    nextEvent(1, true, true).catch(error => toast(error.message));
    reviewOutbox.drain().catch(error => toast(error.message));
  } catch (error) { toast(`尚未保存：${error.message}`); }
}

async function saveNeedsConfirmation() {
  const event = currentSaveEvent(); if (!event) return;
  const payload = {review_outcome: 'needs_confirmation', note: $('#note').value.trim(), compact_response: true,
    expected_revisions: Object.fromEntries(segmentEvents(event).map(item => [item.id, item.review.revision || 0]))};
  try {
    const item = queueOperation('segment', event, payload);
    markLocalPending(item); applyFilters(); toast('待二次确认操作已加入保存队列');
    nextEvent(1, true, true).catch(error => toast(error.message)); reviewOutbox.drain().catch(console.error);
  } catch (error) { toast(`尚未保存：${error.message}`); }
}

async function saveTeamSetup() {
  if (videoLoading || !outboxReady || !state.video || state.video.video_id !== state.currentVideoId) return;
  const profile = state.video.team_profile;
  const payload = {expected_revision: profile.revision, status: 'confirmed',
    teams: {teamA: {hex: $('#teamAColor').value, display_name: $('#teamAName').value.trim()},
      teamB: {hex: $('#teamBColor').value, display_name: $('#teamBName').value.trim()}},
    period_split_sec: $('#periodSplitSec').value, first_period_left_team: $('#firstPeriodLeftTeam').value,
    second_period_left_team: $('#secondPeriodLeftTeam').value};
  try { queueOperation('team', null, payload); toast('队伍设置已加入保存队列'); reviewOutbox.drain().catch(console.error); }
  catch (error) { toast(error.message); }
}

async function undoDecision() {
  const event = currentSaveEvent(); if (!event) return;
  try { queueOperation('undo', event, {expected_revision: event.review.revision}); reviewOutbox.drain().catch(console.error); }
  catch (error) { toast(error.message); }
}
async function saveDecision(status) {
  if (status === 'deleted') state.selectedSegmentLabels.clear();
  return saveMultiLabelSegment();
}
async function saveSegmentDecision(status) { return saveDecision(status); }

async function loadVideo(videoId) {
  setupReliableQueue();
  const generation = ++videoLoadGeneration;
  videoLoadController?.abort(); videoLoadController = new AbortController();
  videoLoading = true; state.currentVideoId = videoId; state.currentEventId = null;
  $('#videoSelect').value = videoId;
  state.video = null; state.pendingAbsoluteSeek = null; state.contextEnd = null;
  const player = $('#player'); player.pause(); player.removeAttribute('src'); player.load();
  $('#videoBadge').textContent = '正在加载…';
  try {
    const video = await api(`/api/videos/${videoId}`, {signal: videoLoadController.signal});
    if (generation !== videoLoadGeneration) return false;
    if (video.video_id !== videoId) throw new Error('视频响应与当前选择不一致');
    state.video = video; state.teamSetupOpen = false;
    state.mediaBaseUrl = video.media_url; $('#videoBadge').textContent = videoId;
    state.viewStart = 0; state.viewEnd = Math.min(600, video.duration_sec || 600);
    overlayPending(); videoLoading = false; renderTeamSetup(); applyFilters();
    const next = video.events.find(event => event.review.status === 'unreviewed') || video.events[0];
    if (next) selectEvent(next.id, true); else renderEvent();
    return true;
  } catch (error) {
    if (generation === videoLoadGeneration) {
      $('#videoBadge').textContent = '视频加载失败，请重新选择'; toast(`加载失败：${error.message}`);
    }
    return false;
  } finally { if (generation === videoLoadGeneration) videoLoading = false; }
}

// Requirements are per selected label, so a save+shot segment needs both
// the save's half and the shot's team, independently.
function attributionRule(label) {
  if (label === 'save') return 'half';
  if (label === 'throw_in') return 'none';
  if (label === 'shot' || (label === 'set_piece' && ['corner', 'free_kick', 'penalty'].some(type => state.secondaryLabels.has(type)))) return 'team';
  return 'legacy';
}

function confirmedAttribution(labels) {
  const values = structuredClone(state.attributionByLabel || {});
  for (const label of labels) {
    const value = values[label] || {};
    const rule = attributionRule(label);
    if (rule === 'half') {
      if (!['left', 'right'].includes(value.field_side)) throw new Error('扑救事件请选择左半场或右半场');
      values[label] = {event_team: 'unknown', field_side: value.field_side, goal_side: 'unknown'};
    } else if (rule === 'team') {
      if (!['teamA', 'teamB'].includes(value.event_team)) throw new Error('射门、角球、任意球、点球事件请选择队伍颜色');
      values[label] = {event_team: value.event_team, field_side: 'unknown', goal_side: label === 'set_piece' ? 'not_applicable' : 'unknown'};
    } else if (rule === 'none') {
      values[label] = {event_team: 'unknown', field_side: 'unknown', goal_side: 'not_applicable'};
    }
  }
  return values;
}

const renderLegacyTeamAttribution = renderTeamAttribution;
renderTeamAttribution = function () {
  renderLegacyTeamAttribution();
  for (const label of state.selectedSegmentLabels) {
    const card = document.querySelector(`#attributionCards .label-${label}`);
    if (!card) continue;
    const rule = attributionRule(label);
    if (rule === 'none') { card.remove(); continue; }
    if (rule === 'legacy') continue;
    card.querySelector('.compact-goal-row')?.remove();
    const hint = card.querySelector('header small');
    if (rule === 'half') {
      card.querySelector('.compact-team-row')?.remove();
      card.querySelector('header strong').textContent = '扑救 · 所在半场';
      card.querySelectorAll('[data-field-side]').forEach(button => {
        if (!['left', 'right'].includes(button.dataset.fieldSide)) button.remove();
      });
      if (hint) hint.textContent = '必选左／右半场';
    } else {
      card.querySelector('.compact-field-row')?.remove();
      card.querySelector('[data-team="unknown"]')?.remove();
      if (hint) hint.textContent = '必选队伍颜色';
    }
  }
};
// Secondary-type handlers rebuild their own controls; update attribution after them.
document.addEventListener('click', event => {
  if (event.target.closest?.('[data-detail]')) queueMicrotask(renderTeamAttribution);
});

// Reuse the demuxer/index on every seek within the same video. Calling load()
// for an unbuffered target discards multi-megabyte MP4 metadata on weak links.
let stableSeekGeneration = 0;
let pendingMetadataSeek = null;
function seekAbsolute(timeSec, autoPlay = false) {
  if (!state.video) return;
  const target = Math.max(0, Math.min(Number(state.video.duration_sec || Infinity), Number(timeSec)));
  if (!Number.isFinite(target)) return;
  const player = $('#player'), videoId = state.currentVideoId;
  const generation = ++stableSeekGeneration;
  if (pendingMetadataSeek) player.removeEventListener('loadedmetadata', pendingMetadataSeek);
  state.pendingAbsoluteSeek = {videoId, target, autoPlay};
  const apply = () => {
    if (generation !== stableSeekGeneration || videoId !== state.currentVideoId || !state.pendingAbsoluteSeek) return;
    try { player.currentTime = target; } catch (_) { return; }
    state.pendingAbsoluteSeek = null; pendingMetadataSeek = null;
    const seekInput = $('#absoluteSeekInput');
    if (seekInput) seekInput.value = formatTime(target);
    if (autoPlay) player.play().catch(error => {
      if (error.name === 'NotAllowedError') toast('请点击画面开始播放');
    });
  };
  if (player.getAttribute('src') && !player.error && player.readyState >= 1) {
    apply();
    return;
  }
  pendingMetadataSeek = apply;
  player.addEventListener('loadedmetadata', apply, {once: true});
  if (!player.getAttribute('src') || player.error) {
    player.src = mediaUrlAt(target);
    player.load();
  }
}
