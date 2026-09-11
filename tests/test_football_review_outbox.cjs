const test = require('node:test');
const assert = require('node:assert/strict');
const {ReviewOutbox} = require('../tools/football_event_review/static/review_outbox.js');
class Storage {
  constructor() { this.data = new Map(); }
  get length() { return this.data.size; }
  key(i) { return [...this.data.keys()][i]; }
  getItem(key) { return this.data.get(key) ?? null; }
  setItem(key, value) { this.data.set(key, value); }
  removeItem(key) { this.data.delete(key); }
}
const job = (id = 'operation-1') => ({id, key: 'v0:s0', payload: {operation_id: id}, videoId: 'v0'});
test('offline record survives a new instance and retries the identical operation', async () => {
  const storage = new Storage();
  const first = new ReviewOutbox({storage, scope: 'task:user', send: async () => {throw new Error('offline');}});
  first.add(job()); await first.drain(); first.stop(); assert.equal(first.list()[0].status, 'retry');
  let sent;
  const next = new ReviewOutbox({storage, scope: 'task:user', send: async item => {sent = item; return {};}});
  next.retry('operation-1'); await new Promise(resolve => setTimeout(resolve, 20)); next.stop();
  assert.equal(sent.payload.operation_id, 'operation-1'); assert.equal(next.list().length, 0);
});
test('a 409 preserves its draft and requires explicit editing before a new operation', async () => {
  const queue = new ReviewOutbox({storage: new Storage(), scope: 'task:user', send: async () => {throw Object.assign(new Error('stale'),{status:409});}});
  queue.add(job()); await queue.drain(); queue.stop();
  assert.equal(queue.list()[0].status,'conflict'); assert.throws(() => queue.add(job('new')));
  queue.edit('operation-1');queue.add(job('new'));assert.equal(queue.list()[0].id,'new');
});
test('quota errors reject before the operation is acknowledged', () => {
  const storage = new Storage();storage.setItem = () => {throw new Error('quota');};
  const queue = new ReviewOutbox({storage,scope:'task:user',send: async () => ({})});
  assert.throws(() => queue.add(job()),/quota/);assert.equal(queue.list().length,0);
});
test('queued data is isolated by both task and authenticated user', () => {
  const storage=new Storage();const a=new ReviewOutbox({storage,scope:'a:user',send:async()=>({})});a.add(job());
  for (const scope of ['a:other','b:user']) assert.equal(new ReviewOutbox({storage,scope,send:async()=>({})}).list().length,0);
});
test('queue length is bounded and a blocked item does not block another segment', async () => {
  const queue=new ReviewOutbox({storage:new Storage(),scope:'a:u',send:async item=>{
    if(item.id==='0')throw Object.assign(new Error('invalid'),{status:400});return {};
  }});
  for(let i=0;i<100;i++)queue.add({...job(String(i)),key:String(i)});
  assert.throws(()=>queue.add({...job('101'),key:'101'}),/100/);
  await queue.drain();queue.stop();assert.equal(queue.list().length,1);assert.equal(queue.list()[0].status,'failed');
});
