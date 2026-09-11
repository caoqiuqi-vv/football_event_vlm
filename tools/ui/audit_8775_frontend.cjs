// Exercise the actual generated loadVideo function with reversed API responses.
// This is a diagnostic with DOM stubs, not a browser playback test.
const fs = require('node:fs');
const vm = require('node:vm');
if (!process.argv[2]) throw new Error('Usage: node audit_8775_frontend.cjs GENERATED_JS');
const source = fs.readFileSync(process.argv[2], 'utf8');
const start = source.lastIndexOf('async function loadVideo(videoId) {');
const legacyEnd = source.indexOf('\nfunction setupTimeline()', start);
const end = legacyEnd >= 0 ? legacyEnd : source.indexOf('\ninit().catch(', start);
if (start < 0 || end < 0) throw new Error('loadVideo boundaries changed; inspect generated JS');
const pending = new Map();
const player = {pause() {}, removeAttribute() {}, load() {}};
const badge = {};
const context = vm.createContext({
  state: {}, videoLoadGeneration: 0, videoLoadController: null, videoLoading: false, AbortController,
  setupReliableQueue() {}, overlayPending() {}, toast() {},
  api: path => new Promise(resolve => pending.set(path, resolve)),
  $: selector => selector === '#player' ? player : badge,
  renderTeamSetup() {}, applyFilters() {}, selectEvent() {}, renderEvent() {},
});
vm.runInContext(source.slice(start, end), context);
(async () => {
  const a = context.loadVideo('A');
  const b = context.loadVideo('B');
  pending.get('/api/videos/B')({video_id: 'B', media_url: '/media/B', duration_sec: 100, events: []});
  await b;
  pending.get('/api/videos/A')({video_id: 'A', media_url: '/media/A', duration_sec: 100, events: []});
  await a;
  console.log(JSON.stringify({
    scenario: 'select A, then B; B response arrives first',
    selected_video: context.state.currentVideoId,
    payload_video: context.state.video.video_id,
    media_url: context.state.mediaBaseUrl,
    mismatch: context.state.currentVideoId !== context.state.video.video_id,
  }, null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; });
