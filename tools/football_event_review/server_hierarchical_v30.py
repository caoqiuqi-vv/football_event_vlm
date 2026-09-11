#!/usr/bin/env python3
"""Evidence-safe review playback and deletion guards."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v29  # noqa: F401


_html_v29 = hierarchical.patch_hierarchical_html
_js_v29 = hierarchical.patch_hierarchical_js


def patch_html_v30(text: str) -> str:
    return _html_v29(text).replace(
        "hierarchical-v29-throw-in-label-20260902",
        "hierarchical-v30-evidence-safe-review-20260902",
        1,
    )


def patch_js_v30(text: str) -> str:
    text = _js_v29(text)
    play_anchor = "function playEventContext(event) {"
    helpers = r'''let reviewedGtAnchors = new Set();
let reviewedGtSegmentKey = null;

function gtAnchorsForEvent(event) {
  return [...new Set((event?.matching_gt_times || []).map(Number).filter(Number.isFinite))];
}

function segmentEvidenceBounds(event = currentEvent()) {
  const group = segmentEvents(event);
  const starts = group.map((item) => Number(item.support_start_sec ?? item.start_sec ?? item.time_sec));
  const ends = group.map((item) => Number(item.support_end_sec ?? item.end_sec ?? item.time_sec));
  const gtTimes = group.flatMap(gtAnchorsForEvent);
  if (gtTimes.length) {
    starts.push(Math.min(...gtTimes) - 3);
    ends.push(Math.max(...gtTimes) + 5);
  }
  return {
    start: Math.max(0, Math.min(...starts.filter(Number.isFinite))),
    end: Math.min(Number(state.video?.duration_sec || Infinity), Math.max(...ends.filter(Number.isFinite))),
  };
}

function gtAnchorKey(event, timeSec) {
  return `${event.id}@${Number(timeSec).toFixed(3)}`;
}

function updateReviewedGtAnchors() {
  const event = currentEvent();
  const player = $("#player");
  if (!event || !player) return;
  for (const item of segmentEvents(event)) {
    for (const timeSec of gtAnchorsForEvent(item)) {
      if (Math.abs(Number(player.currentTime) - timeSec) <= 1.25) {
        reviewedGtAnchors.add(gtAnchorKey(item, timeSec));
      }
    }
  }
}

function guardGtDeletion(selected) {
  const event = currentEvent();
  if (!event) return true;
  const missing = [];
  for (const item of segmentEvents(event)) {
    if (selected.has(item.label)) continue;
    for (const timeSec of gtAnchorsForEvent(item)) {
      if (!reviewedGtAnchors.has(gtAnchorKey(item, timeSec))) missing.push({ item, timeSec });
    }
  }
  if (!missing.length) return true;
  const first = missing.sort((a, b) => a.timeSec - b.timeSec)[0];
  const player = $("#player");
  player.currentTime = Math.max(0, first.timeSec - 3);
  state.contextEnd = Math.min(state.video.duration_sec, first.timeSec + 5);
  player.play().catch(() => {});
  toast(`该操作会删除 GT ${LABEL_NAMES[first.item.label] || first.item.label} @ ${formatTime(first.timeSec)}；已跳到原时间，请看完后再次确认`);
  return false;
}

'''
    if play_anchor not in text:
        raise RuntimeError("v30 play function anchor missing")
    text = text.replace(play_anchor, helpers + play_anchor, 1)

    play_start = text.index("function playEventContext(event) {")
    play_end = text.index("function playFullSegment(event) {", play_start)
    safe_play = r'''function playEventContext(event) {
  const player = $("#player");
  const bounds = segmentEvidenceBounds(event);
  player.currentTime = bounds.start;
  state.contextEnd = bounds.end;
  const playback = player.play();
  if (playback) playback.catch(() => toast("浏览器阻止了自动播放，请点击画面或按 Space"));
}

'''
    text = text[:play_start] + safe_play + text[play_end:]

    selected_anchor = "  const selected = new Set(selectedLabels);"
    if selected_anchor not in text:
        raise RuntimeError("v30 save guard anchor missing")
    text = text.replace(selected_anchor, selected_anchor + "\n  if (!guardGtDeletion(selected)) return;", 1)

    select_anchor = '''  const event = currentEvent();
  if (event) showSwitchFeedback(event);
  if (event && seek) {'''
    select_replacement = '''  const event = currentEvent();
  if (event && reviewedGtSegmentKey !== segmentKey(event)) {
    reviewedGtSegmentKey = segmentKey(event);
    reviewedGtAnchors = new Set();
  }
  if (event) showSwitchFeedback(event);
  if (event && seek) {'''
    if select_anchor not in text:
        raise RuntimeError("v30 select reset anchor missing")
    text = text.replace(select_anchor, select_replacement, 1)
    old_seek = '    else $("#player").currentTime = Math.max(0, event.time_sec - 2.5);'
    new_seek = '    else $("#player").currentTime = segmentEvidenceBounds(event).start;'
    if old_seek not in text:
        raise RuntimeError("v30 select seek anchor missing")
    text = text.replace(old_seek, new_seek, 1)

    old_range = '  $("#supportRange").textContent = `片段 ${formatTime(Math.min(...group.map((item) => item.support_start_sec ?? item.start_sec)))} – ${formatTime(Math.max(...group.map((item) => item.support_end_sec ?? item.end_sec)))}`;'
    new_range = '''  const gtBadges = group.flatMap((item) => gtAnchorsForEvent(item).map((timeSec) => `${LABEL_NAMES[item.label] || item.label} ${formatTime(timeSec)}`));
  $("#supportRange").textContent = `片段 ${formatTime(Math.min(...group.map((item) => item.support_start_sec ?? item.start_sec)))} – ${formatTime(Math.max(...group.map((item) => item.support_end_sec ?? item.end_sec)))}${gtBadges.length ? ` · GT证据：${gtBadges.join(" / ")}` : ""}`;'''
    if old_range not in text:
        raise RuntimeError("v30 evidence label anchor missing")
    text = text.replace(old_range, new_range, 1)

    old_timeupdate = 'player.ontimeupdate = () => { $("#timeBadge").textContent = formatTime(player.currentTime); if (state.contextEnd && player.currentTime >= state.contextEnd) { player.pause(); state.contextEnd = null; } scheduleTimelineDraw(); };'
    new_timeupdate = 'player.ontimeupdate = () => { $("#timeBadge").textContent = formatTime(player.currentTime); updateReviewedGtAnchors(); if (state.contextEnd && player.currentTime >= state.contextEnd) { player.pause(); state.contextEnd = null; } scheduleTimelineDraw(); };'
    if old_timeupdate not in text:
        raise RuntimeError("v30 timeupdate anchor missing")
    return text.replace(old_timeupdate, new_timeupdate, 1)


hierarchical.patch_hierarchical_html = patch_html_v30
hierarchical.patch_hierarchical_js = patch_js_v30


if __name__ == "__main__":
    hierarchical.main()
