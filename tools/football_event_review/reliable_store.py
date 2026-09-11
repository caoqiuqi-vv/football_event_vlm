"""Transactional v40 persistence, operation receipts and read-only authentication."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import secrets
import sqlite3
import threading
import time
from urllib.parse import urlencode

import server_multiuser_v30 as v30
import server_multiuser_v36 as v36
from mp4_layout import is_faststart
from attribution_rules import normalize_segment_attribution


class ReliableReviewStore(v36.TeamCalibrationStore):
    def __init__(self, *args):
        self._local = threading.local()
        self._wal_ready = False
        self._activity_lock = threading.Lock()
        self._activity = {}
        self._layout_cache = {}
        super().__init__(*args)
        with self.connect() as connection:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS review_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS review_operations(
                    user_id TEXT NOT NULL, operation_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    result_json TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, operation_id));
                CREATE TABLE IF NOT EXISTS review_segment_index(
                    event_id TEXT PRIMARY KEY REFERENCES events(id),
                    video_id TEXT NOT NULL, segment_id TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS review_segment_lookup
                    ON review_segment_index(video_id, segment_id);
            ''')
            connection.execute('INSERT OR IGNORE INTO review_metadata VALUES (?, ?)',
                               ('namespace', secrets.token_hex(16)))
            self.namespace = connection.execute(
                "SELECT value FROM review_metadata WHERE key='namespace'").fetchone()[0]
            rows = connection.execute('SELECT id,video_id,payload_json FROM events').fetchall()
            connection.executemany('INSERT OR REPLACE INTO review_segment_index VALUES (?,?,?)',
                [(r['id'], r['video_id'], str(json.loads(r['payload_json']).get('segment_id') or r['id']))
                 for r in rows])

    def _open_connection(self, timeout=5):
        connection = sqlite3.connect(self.db_path, timeout=timeout)
        connection.row_factory = sqlite3.Row
        if not self._wal_ready:
            connection.execute('PRAGMA journal_mode=WAL')
            self._wal_ready = True
        connection.execute('PRAGMA foreign_keys=ON')
        return connection

    @contextmanager
    def connect(self):
        # All legacy nested `with self.connect()` calls participate in the outer
        # operation. No inner context may commit, roll back or close that connection.
        shared = getattr(self._local, 'connection', None)
        if shared is not None:
            yield shared
            return
        connection = self._open_connection()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self):
        if getattr(self._local, 'connection', None) is not None:
            yield self._local.connection
            return
        with self.lock:
            connection = self._open_connection()
            try:
                connection.execute('BEGIN IMMEDIATE')
                self._local.connection = connection
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                self._local.connection = None
                self._local.segment = None
                connection.close()

    def _operation(self, user, kind, target, payload, action):
        operation_id = payload.get('operation_id')
        if not isinstance(operation_id, str) or not 8 <= len(operation_id) <= 128:
            raise ValueError('Missing or invalid operation_id; refresh the review page')
        fingerprint = hashlib.sha256(json.dumps(
            [kind, target, payload], sort_keys=True, allow_nan=False,
            separators=(',', ':')).encode()).hexdigest()
        with self.transaction() as connection:
            video_id = target if kind == 'team-profile' else self.event_video_id(target)
            if not self.owns_video(str(user['user_id']), video_id):
                raise PermissionError(video_id)
            previous = connection.execute(
                'SELECT fingerprint,result_json FROM review_operations WHERE user_id=? AND operation_id=?',
                (user['user_id'], operation_id)).fetchone()
            if previous:
                if previous['fingerprint'] != fingerprint:
                    raise v30.RevisionConflict('operation_id reused with different content')
                return json.loads(previous['result_json'])
            result = action()
            result['operation_id'] = operation_id
            connection.execute('INSERT INTO review_operations VALUES (?,?,?,?,?)',
                (user['user_id'], operation_id, fingerprint,
                 json.dumps(result, ensure_ascii=False, allow_nan=False), v30.utc_now()))
            return result

    def update_segment_for_user(self, user, event_id, payload):
        parent = super().update_segment_for_user
        def action():
            anchor = self._event_by_id(event_id)
            if 'corrected_time_sec' in payload:
                value = float(payload['corrected_time_sec'])
                if not math.isfinite(value) or not 0 <= value <= float(self.videos[anchor['video_id']]['duration_sec']):
                    raise ValueError('Event time outside video duration')
            self._local.segment = (anchor['video_id'], str(anchor.get('segment_id') or event_id))
            return parent(user, event_id, normalize_segment_attribution(payload))
        return self._operation(user, 'segment-decision', event_id, payload, action)

    def events_for_video(self, video_id):
        scope = getattr(self._local, 'segment', None)
        if scope and scope[0] == video_id:
            with self.connect() as connection:
                rows = connection.execute(self._event_select('''
                    JOIN review_segment_index si ON si.event_id=e.id
                    WHERE si.video_id=? AND si.segment_id=?
                    ORDER BY e.source_time_sec,e.source_label'''), scope).fetchall()
            return [self._event_from_row(row) for row in rows]
        return super().events_for_video(video_id)

    def _insert_human_label(self, anchor, label):
        event_id = super()._insert_human_label(anchor, label)
        with self.connect() as connection:
            connection.execute('INSERT OR REPLACE INTO review_segment_index VALUES (?,?,?)',
                (event_id, anchor['video_id'], str(anchor.get('segment_id') or anchor['id'])))
        return event_id

    def update_team_profile_for_user(self, user, video_id, data):
        parent = super().update_team_profile_for_user
        def action():
            split = data.get('period_split_sec')
            if split not in (None, ''):
                if not math.isfinite(float(split)) or not 0 <= float(split) <= float(self.videos[video_id]['duration_sec']):
                    raise ValueError('Invalid period split time')
            return parent(user, video_id, data)
        return self._operation(user, 'team-profile', video_id, data, action)

    def undo_for_user(self, user, event_id, data):
        def action():
            current = self._event_by_id(event_id)
            revision = int(current['review']['revision'])
            if data.get('expected_revision') != revision:
                raise v30.RevisionConflict('stale undo revision')
            super(ReliableReviewStore, self).undo(event_id)
            with self.connect() as connection:
                # Revisions remain monotonic even when historical content is restored.
                connection.execute('UPDATE reviews SET revision=?,reviewer=?,updated_at=? WHERE event_id=?',
                    (revision + 1, user['display_name'], v30.utc_now(), event_id))
            result = self._event_by_id(event_id)
            self._write_audit(request_id=secrets.token_hex(16), user_id=user['user_id'],
                video_id=current['video_id'], segment_id=str(current.get('segment_id') or event_id),
                action='undo', payload=data, result=result, status='committed')
            return result
        return self._operation(user, 'undo', event_id, data, action)

    def session_user(self, raw_token):
        if not raw_token:
            return None
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        current = time.time()
        with self.connect() as connection:
            row = connection.execute('''SELECT u.user_id,u.slug,u.display_name,s.expires_at
                FROM review_sessions s JOIN review_users u ON u.user_id=s.user_id
                WHERE s.token_hash=? AND s.expires_at>? AND u.active=1''', (token_hash, current)).fetchone()
        if row is None:
            return None
        with self._activity_lock:
            if len(self._activity) < 4096 or token_hash in self._activity:
                self._activity[token_hash] = current
        return dict(row)

    def flush_session_activity(self):
        with self._activity_lock:
            pending, self._activity = self._activity, {}
        if not pending:
            return
        connection = self._open_connection(timeout=0)
        try:
            with connection:
                connection.executemany('''UPDATE review_sessions SET last_seen_at=?
                    WHERE token_hash=? AND last_seen_at < ?''',
                    [(stamp, key, stamp - 60) for key, stamp in pending.items()])
        except sqlite3.OperationalError:
            # Telemetry may be delayed; it must never delay authentication/media.
            with self._activity_lock:
                for key, stamp in pending.items():
                    if len(self._activity) < 4096:
                        self._activity[key] = max(stamp, self._activity.get(key, 0))
        finally:
            connection.close()

    def bootstrap_for_user(self, user):
        result = super().bootstrap_for_user(user)
        result['review_namespace'] = self.namespace
        result['reliable_save_version'] = 1
        return result

    @staticmethod
    def media_version(path, st=None):
        st = st or path.stat()
        return hashlib.sha256(f'{path.resolve()}:{st.st_size}:{st.st_mtime_ns}'.encode()).hexdigest()[:24]

    def video_payload_for_user(self, user_id, video_id):
        result = super().video_payload_for_user(user_id, video_id)
        path, variant = self.media_path(video_id)
        if path.is_file():
            query = {'v': self.media_version(path)}
            if variant == 'original':
                query['quality'] = 'original'
            result['media_url'] = f'/media/{video_id}?' + urlencode(query)
        return result

    def media_readiness(self):
        faststart = missing = 0
        for video_id in self.videos:
            path, _ = self.media_path(video_id)
            if not path.is_file():
                missing += 1
                continue
            version = self.media_version(path)
            cached = self._layout_cache.get(video_id)
            if cached is None or cached[0] != version:
                cached = (version, is_faststart(path))
                self._layout_cache[video_id] = cached
            faststart += int(cached[1])
        return {'total': len(self.videos), 'faststart': faststart, 'missing': missing,
                'ready': missing == 0 and faststart == len(self.videos)}
