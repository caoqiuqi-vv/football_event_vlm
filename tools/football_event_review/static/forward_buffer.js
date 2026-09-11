// Native forward buffering shares the current player's index, connection and cache.
// preload is a browser hint, not a hard network/memory quota. Never seek/load to warm it.
function forwardBufferPolicy({ready, seeking, failed, hidden, saveData, effectiveType, downlink,
                              current, bufferedEnd, next}) {
  if (failed || hidden || seeking || ready < 3 || saveData ||
      ['slow-2g', '2g', '3g'].includes(effectiveType) ||
      (Number.isFinite(downlink) && downlink > 0 && downlink < 2)) return 'metadata';
  if (!Number.isFinite(next) || next < current || next - current > 30) return 'metadata';
  return bufferedEnd < next + 8 ? 'auto' : 'metadata';
}
function installForwardBuffer(player, nextTarget) {
  let blockedUntil = 0;
  const update = () => {
    const network = navigator.connection || {};
    let end = player.currentTime;
    for (let i = 0; i < player.buffered.length; i++) {
      if (player.buffered.start(i) <= player.currentTime && player.buffered.end(i) >= player.currentTime) {
        end = player.buffered.end(i); break;
      }
    }
    const mode = forwardBufferPolicy({ready: player.readyState, seeking: player.seeking,
      failed: player.error || Date.now() < blockedUntil, hidden: document.hidden,
      saveData: network.saveData, effectiveType: network.effectiveType, downlink: network.downlink,
      current: player.currentTime, bufferedEnd: end, next: nextTarget()});
    if (player.preload !== mode) player.preload = mode;
  };
  for (const name of ['waiting', 'stalled']) player.addEventListener(name, () => {
    blockedUntil = Date.now() + 30000; player.preload = 'metadata';
  });
  for (const name of ['progress', 'canplay', 'seeked', 'pause', 'timeupdate', 'seeking', 'emptied', 'error']) {
    player.addEventListener(name, update);
  }
  document.addEventListener('visibilitychange', update);
  navigator.connection?.addEventListener('change', update);
  update();
}
