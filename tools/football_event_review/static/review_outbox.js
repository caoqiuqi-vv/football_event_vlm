/* Durable per-operation records. Server receipts make duplicate delivery harmless. */
(function (root) {
  class ReviewOutbox {
    constructor({storage, scope, send, onChange = () => {}, onSaved = () => {}, now = Date.now}) {
      this.storage = storage; this.prefix = `football-review-v40:${scope}:`;
      this.send = send; this.onChange = onChange; this.onSaved = onSaved; this.now = now;
      this.running = false; this.pausedAuth = false; this.timer = null;
    }
    list() {
      const items = [];
      for (let i = 0; i < this.storage.length; i++) {
        const key = this.storage.key(i);
        if (key?.startsWith(this.prefix)) {
          const item = JSON.parse(this.storage.getItem(key));
          if (item && item.id && item.payload) items.push(item);
        }
      }
      return items.sort((a, b) => a.created - b.created || a.id.localeCompare(b.id));
    }
    write(item) { this.storage.setItem(this.prefix + item.id, JSON.stringify(item)); this.onChange(); }
    add(item) {
      const items = this.list();
      if (items.length >= 100) throw new Error('待保存操作已达 100 条，请先恢复连接并处理保存队列');
      if (items.some(old => old.key === item.key && old.status !== 'editing'))
        throw new Error('此片段还有待保存操作，请先在保存队列中处理');
      const record = JSON.parse(JSON.stringify({...item, created: this.now(), status: 'pending', attempts: 0, nextAttempt: 0}));
      this.write(record); // synchronous durable acknowledgement BEFORE changing the UI
      for (const old of items.filter(old => old.key === item.key && old.status === 'editing'))
        this.storage.removeItem(this.prefix + old.id);
      this.onChange();
      return record;
    }
    retry(id) {
      const item = this.list().find(x => x.id === id);
      if (!item || ['conflict', 'failed', 'editing'].includes(item.status)) return;
      this.pausedAuth = false; item.nextAttempt = 0; item.status = 'pending'; this.write(item); this.drain();
    }
    edit(id) {
      const item = this.list().find(x => x.id === id);
      if (!item || !['conflict', 'failed', 'editing'].includes(item.status))
        throw new Error('发送中的操作不能改写，请等待保存结果');
      item.status = 'editing'; this.write(item); return item;
    }
    discard(id) {
      const item = this.list().find(x => x.id === id);
      if (!item || !['conflict', 'failed', 'editing'].includes(item.status)) throw new Error('仅可放弃明确失败或冲突的草稿');
      this.storage.removeItem(this.prefix + id); this.onChange();
    }
    async drain() {
      if (this.running || this.pausedAuth) return;
      this.running = true;
      try {
        for (;;) {
          const item = this.list().find(x => ['pending', 'retry'].includes(x.status) && x.nextAttempt <= this.now());
          if (!item) break;
          let result;
          try {
            result = await this.send(item);
          } catch (error) {
            item.attempts += 1;
            item.error = error.message || '网络连接失败';
            if (error.status === 401) {
              item.status = 'auth'; this.pausedAuth = true;
            } else if (error.status === 409) item.status = 'conflict';
            else if (error.status >= 400 && error.status < 500 && ![408, 429].includes(error.status)) item.status = 'failed';
            else { item.status = 'retry'; item.nextAttempt = this.now() + Math.min(10000, 500 * 2 ** Math.min(item.attempts, 5)); }
            this.write(item);
            if (this.pausedAuth) break;
            continue;
          }
          // Leave the receipt recoverable if storage removal fails. Retrying the
          // same operation_id must return the committed result on the server.
          this.storage.removeItem(this.prefix + item.id);
          try { await this.onSaved(item, result); }
          catch (error) { console.error('Saved successfully; UI refresh failed', error); }
          this.onChange();
        }
      } finally {
        this.running = false;
        this.onChange();
        clearTimeout(this.timer);
        const waiting = this.list().filter(x => ['pending', 'retry'].includes(x.status));
        if (waiting.length && !this.pausedAuth) {
          const delay = Math.max(100, Math.min(...waiting.map(x => x.nextAttempt)) - this.now());
          this.timer = setTimeout(() => this.drain().catch(console.error), delay);
        }
      }
    }
    resume() {
      this.pausedAuth = false;
      for (const item of this.list()) {
        if (item.status === 'auth') { item.status = 'pending'; item.nextAttempt = 0; this.write(item); }
      }
      return this.drain();
    }
    stop() { clearTimeout(this.timer); this.pausedAuth = true; }
  }
  root.ReviewOutbox = ReviewOutbox;
  if (typeof module !== 'undefined') module.exports = {ReviewOutbox};
})(globalThis);
