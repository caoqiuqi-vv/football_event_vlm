#!/usr/bin/env python3
"""Make segment navigation safe at video/task boundaries.

The v9 navigation assumed that a next segment always existed.  Saving the
last unreviewed segment therefore passed ``null`` to ``segmentRepresentative``
and left the browser in a broken-looking state.  This patch keeps navigation
within a video when possible and otherwise advances to the next video.  The
"next unreviewed" path wraps once, so an annotator cannot strand older items.
"""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v25  # noqa: F401


_html_v25 = hierarchical.patch_hierarchical_html
_js_v25 = hierarchical.patch_hierarchical_js


def _replace_function(text: str, start_name: str, end_name: str, replacement: str) -> str:
    start = text.find(f"function {start_name}(")
    end = text.find(f"function {end_name}(", start)
    if start < 0 or end < 0:
        raise RuntimeError(f"v26 function anchors missing: {start_name}/{end_name}")
    # The inherited functions are a mixture of sync and async declarations.
    # Include the optional ``async `` prefix in the replaced span; otherwise
    # replacing the preceding function can accidentally strip ``async`` from
    # the following declaration.
    if text[max(0, start - 6):start] == "async ":
        start -= 6
    if text[max(0, end - 6):end] == "async ":
        end -= 6
    return text[:start] + replacement.rstrip() + "\n\n" + text[end:]


def patch_html_v26(text: str) -> str:
    return _html_v25(text).replace(
        "hierarchical-v25-back-pass-label-20260901",
        "hierarchical-v26-boundary-navigation-20260901",
        1,
    )


def patch_js_v26(text: str) -> str:
    text = _js_v25(text)

    navigation = r'''async function navigateAcrossVideos(direction, onlyUnreviewed = false, autoPlay = true) {
  if (!state.bootstrap?.videos?.length || !state.video) return false;
  const videos = state.bootstrap.videos;
  const currentVideoIndex = videos.findIndex((video) => video.video_id === state.currentVideoId);
  const startIndex = currentVideoIndex >= 0 ? currentVideoIndex : 0;
  const candidateIndexes = [];
  if (onlyUnreviewed) {
    // Wrap once for review navigation: the last item in the last video can
    // still reach an older unreviewed segment.
    for (let offset = 1; offset < videos.length; offset += 1) {
      candidateIndexes.push((startIndex + direction * offset + videos.length) % videos.length);
    }
  } else {
    for (let index = startIndex + direction; index >= 0 && index < videos.length; index += direction) {
      candidateIndexes.push(index);
    }
  }

  for (const videoIndex of candidateIndexes) {
    const summary = videos[videoIndex];
    if (onlyUnreviewed && Number(summary.reviewed || 0) >= Number(summary.total || 0)) continue;
    const payload = await api(`/api/videos/${summary.video_id}`);
    const groups = segmentGroups(payload.events || []);
    const ordered = direction > 0 ? groups : [...groups].reverse();
    const targetGroup = onlyUnreviewed
      ? ordered.find((group) => group.some((event) => event.review.status === "unreviewed"))
      : ordered[0];
    if (!targetGroup) continue;
    const targetId = segmentRepresentative(targetGroup)?.id;
    if (!targetId) continue;
    await loadVideo(summary.video_id);
    $("#videoSelect").value = summary.video_id;
    selectEvent(targetId, true, autoPlay);
    return true;
  }

  if (onlyUnreviewed) {
    // All other videos were searched.  Finish the wrap by checking the part
    // of the current video on the other side of the current segment.
    const groups = segmentGroups();
    const currentKey = segmentKey(currentEvent());
    const currentIndex = groups.findIndex((group) => segmentKey(group[0]) === currentKey);
    const wrapped = direction > 0
      ? groups.slice(0, Math.max(0, currentIndex))
      : groups.slice(currentIndex + 1).reverse();
    const targetGroup = wrapped.find((group) => group.some((event) => event.review.status === "unreviewed"));
    const targetId = targetGroup ? segmentRepresentative(targetGroup)?.id : null;
    if (targetId) {
      selectEvent(targetId, true, autoPlay);
      return true;
    }
    toast("全部片段均已审核完成");
  } else {
    toast(direction > 0 ? "已经是任务最后一个片段" : "已经是任务第一个片段");
  }
  return false;
}

async function nextEvent(direction = 1, onlyUnreviewed = false, autoPlay = true) {
  if (!state.video) return;
  const allGroups = segmentGroups();
  const currentKey = segmentKey(currentEvent());
  const currentIndex = allGroups.findIndex((group) => segmentKey(group[0]) === currentKey);
  let targetGroup = null;
  if (onlyUnreviewed) {
    const candidates = direction > 0
      ? allGroups.slice(currentIndex + 1)
      : allGroups.slice(0, Math.max(0, currentIndex)).reverse();
    targetGroup = candidates.find((group) => group.some((event) => event.review.status === "unreviewed")) || null;
  } else {
    targetGroup = allGroups[currentIndex < 0 ? 0 : currentIndex + direction] || null;
  }
  const targetId = targetGroup ? segmentRepresentative(targetGroup)?.id : null;
  if (targetId) {
    selectEvent(targetId, true, autoPlay);
    return;
  }
  await navigateAcrossVideos(direction, onlyUnreviewed, autoPlay);
}'''
    text = _replace_function(text, "nextEvent", "navigateAdjacent", navigation)

    adjacent = r'''async function navigateAdjacent(direction) {
  if (!state.video) return;
  const groups = segmentGroups();
  const currentKey = segmentKey(currentEvent());
  const index = groups.findIndex((group) => segmentKey(group[0]) === currentKey);
  const target = groups[index + direction];
  const targetId = target ? segmentRepresentative(target)?.id : null;
  if (targetId) {
    selectEvent(targetId, true, true);
    return;
  }
  await navigateAcrossVideos(direction, false, true);
}'''
    text = _replace_function(text, "navigateAdjacent", "saveDecision", adjacent)
    return text


hierarchical.patch_hierarchical_html = patch_html_v26
hierarchical.patch_hierarchical_js = patch_js_v26


if __name__ == "__main__":
    hierarchical.main()
