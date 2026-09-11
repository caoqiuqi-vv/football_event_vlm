from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import http.client
import json
from pathlib import Path
import secrets
import shutil
import socket
import ssl
import subprocess
import sqlite3
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

TOOL = Path(__file__).resolve().parents[1] / 'tools/football_event_review'
sys.path.insert(0, str(TOOL))
import server_multiuser_v40 as v40
import server_multiuser_v30 as v30


class ReliableReviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        media = self.root / 'video.mp4'; media.write_bytes(bytes(range(256)) * 8192)
        videos = []
        for i in range(3):
            video_id = f'v{i}'
            events = [dict(id=f'{video_id}-e{j}', video_id=video_id,
                segment_id=f'{video_id}-s{j}', label='shot', score=.9,
                time_sec=10+j*5, start_sec=5+j*5, end_sec=15+j*5) for j in range(40)]
            videos.append(dict(video_id=video_id, video_path=str(media), duration_sec=300, events=events))
        salt = '01' * 16
        digest = v30.MultiUserReviewStore.password_digest('test-only', salt)
        self.users = [dict(user_id=f'u{i}', slug=f'u{i}', display_name=f'User {i}',
            password_salt=salt, password_hash=digest, video_ids=['v0','v2'] if i==0 else ['v1']) for i in range(2)]
        self.manifest = self.root/'manifest.json'; self.manifest.write_text(json.dumps(dict(videos=videos)))
        self.access = self.root/'access.json'; self.access.write_text(json.dumps(dict(users=self.users)))
        self.store = self.new_store()
        class Quiet(v40.ReliableReviewHandler):
            def log_message(self, *args): pass
        self.server = v40.ReliableHTTPServer(('127.0.0.1',0), Quiet)
        self.server.store = self.store; self.server.is_tls = False
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.token = self.store.create_session('u0')
        self.cookie = f'{v30.SESSION_COOKIE}={self.token}'

    def new_store(self):
        return v40.ReliableReviewStore(self.manifest,self.root/'db.sqlite3',self.access,self.root/'proxy')

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2); self.temp.cleanup()

    def payload(self, event_id='v0-e0', labels=None):
        anchor=self.store._event_by_id(event_id)
        group=[e for e in self.store.events_for_video(anchor['video_id']) if e.get('segment_id')==anchor.get('segment_id')]
        return dict(operation_id=secrets.token_hex(16), selected_labels=labels or ['shot'],
            active_label='shot', corrected_time_sec=10, compact_response=True,
            attribution_by_label={'shot':{'event_team':'teamA'}, 'save':{'field_side':'left'}},
            expected_revisions={e['id']:e['review']['revision'] for e in group})

    def snapshot(self):
        with self.store.connect() as c:
            return {table:[tuple(r) for r in c.execute(f'SELECT * FROM {table} ORDER BY 1')]
                for table in ('events','reviews','review_history','submission_audit','review_operations','review_segment_index')}

    def request(self,path,method='GET',data=None,headers=None,connection=None):
        owned=connection is None
        c=connection or http.client.HTTPConnection(*self.server.server_address,timeout=2)
        h={'Cookie':self.cookie, **(headers or {})}
        body=json.dumps(data) if data is not None else None
        try:
            c.request(method,path,body=body,headers=h);r=c.getresponse();content=r.read()
            return r.status,dict(r.getheaders()),content
        finally:
            if owned:c.close()

    def test_media_small_writes_restore_timeout_and_release_slot_after_disconnect(self):
        from types import SimpleNamespace
        for fail in (False, True):
            timeouts = []
            connection = SimpleNamespace(gettimeout=lambda: 15, settimeout=timeouts.append)
            chunks = []
            def send(chunk):
                chunks.append(len(chunk))
                if fail:
                    raise TimeoutError('slow client')
            handler = object.__new__(v40.ReliableReviewHandler)
            handler.server = self.server
            handler.connection = connection
            handler.path = '/media/v0'
            handler.headers = {}
            handler.command = 'GET'
            handler.wfile = SimpleNamespace(write=send)
            handler.send_response = lambda *a: None
            headers = {}
            handler.send_header = lambda k, v: headers.update({k: v})
            handler.end_headers = lambda: None
            handler.log_message = lambda *a: None
            handler.send_media('v0')
            self.assertEqual(timeouts, [45, 15])
            self.assertLessEqual(max(chunks), 32768)
            if not fail:
                self.assertEqual(sum(chunks), 2097152)
            self.assertIn('private', headers['Cache-Control'])
            self.assertEqual(headers['Vary'], 'Cookie')
            self.assertTrue(self.server.media_slots.acquire(blocking=False))
            self.server.media_slots.release()

    def test_late_validation_rolls_back_every_table(self):
        before=self.snapshot();data=self.payload(labels=['shot','back_pass'])
        data['attribution_by_label']={'shot':{'event_team':'teamA'},'back_pass':{'field_side':'INVALID'}}
        with self.assertRaises(ValueError):self.store.update_segment_for_user(self.users[0],'v0-e0',data)
        self.assertEqual(before,self.snapshot())

    def test_audit_failure_rolls_back_and_retry_can_commit(self):
        before=self.snapshot();data=self.payload()
        with patch.object(self.store,'_write_audit',side_effect=sqlite3.OperationalError('injected')):
            with self.assertRaises(sqlite3.OperationalError):self.store.update_segment_for_user(self.users[0],'v0-e0',data)
        self.assertEqual(before,self.snapshot())
        self.store.update_segment_for_user(self.users[0],'v0-e0',data)
        self.assertEqual(self.store._event_by_id('v0-e0')['review']['revision'],1)

    def test_lost_response_retry_is_exactly_once(self):
        data=self.payload();first=self.store.update_segment_for_user(self.users[0],'v0-e0',data)
        before=self.snapshot();second=self.store.update_segment_for_user(self.users[0],'v0-e0',data)
        self.assertEqual(first,second);self.assertEqual(before,self.snapshot())
        data['note']='different'
        with self.assertRaises(v30.RevisionConflict):self.store.update_segment_for_user(self.users[0],'v0-e0',data)
        self.assertEqual(before,self.snapshot())

    def test_two_store_instances_reject_stale_revision(self):
        other=self.new_store();one=self.payload();two={**one,'operation_id':secrets.token_hex(16)}
        barrier=threading.Barrier(2)
        def run(args):
            store,payload=args;barrier.wait()
            try:store.update_segment_for_user(self.users[0],'v0-e0',payload);return 'saved'
            except v30.RevisionConflict:return 'conflict'
        with ThreadPoolExecutor(2) as pool:results=list(pool.map(run,[(self.store,one),(other,two)]))
        self.assertEqual(sorted(results),['conflict','saved'])

    def test_duplicate_concurrent_operation_returns_same_receipt(self):
        data=self.payload()
        with ThreadPoolExecutor(4) as pool:
            results=list(pool.map(lambda _: self.store.update_segment_for_user(self.users[0],'v0-e0',data),range(4)))
        self.assertTrue(all(r==results[0] for r in results));self.assertEqual(self.store._event_by_id('v0-e0')['review']['revision'],1)

    def test_multi_label_results_are_segment_local(self):
        result=self.store.update_segment_for_user(self.users[0],'v0-e0',self.payload(labels=['shot','save']))
        self.assertEqual({e['label'] for e in result['segment_events']},{'shot','save'})
        self.assertEqual(len(self.store.events_for_video('v0')),41)
        self.assertEqual(self.store._event_by_id('v0-e1')['review']['revision'],0)

    def test_event_specific_requirements_reject_incomplete_confirmation_atomically(self):
        for label, detail, value in [('save', None, {}), ('save', None, {'field_side':'middle'}),
                ('shot', None, {}), ('set_piece','corner',{}),
                ('set_piece','free_kick',{}), ('set_piece','penalty',{'event_team':'unknown'})]:
            with self.subTest(label=label, detail=detail, value=value):
                data=self.payload(labels=[label]);data['attribution_by_label']={label:value}
                if detail:data['secondary_labels_by_label']={'set_piece':[detail]}
                before=self.snapshot()
                with self.assertRaises(ValueError):self.store.update_segment_for_user(self.users[0],'v0-e0',data)
                self.assertEqual(before,self.snapshot())

    def test_event_specific_confirmation_clears_hidden_fields(self):
        cases=[('save',None,{'field_side':'right','event_team':'teamB','goal_side':'left'},'unknown','right'),
            ('throw_in',None,{'field_side':'left','event_team':'teamB','goal_side':'right'},'unknown','unknown'),
            ('shot',None,{'event_team':'teamB','field_side':'left','goal_side':'right'},'teamB','unknown')]
        cases += [('set_piece',detail,{'event_team':'teamA','field_side':'right'},'teamA','unknown')
                  for detail in ['corner','free_kick','penalty']]
        for i,(label,detail,value,team,half) in enumerate(cases):
            with self.subTest(label=label,detail=detail):
                event_id=f'v0-e{i}';data=self.payload(event_id,labels=[label]);data['attribution_by_label']={label:value}
                if detail:data['secondary_labels_by_label']={'set_piece':[detail]}
                result=self.store.update_segment_for_user(self.users[0],event_id,data)
                saved=next(e for e in result['segment_events'] if e['label']==label)['review']
                self.assertEqual(saved['event_team'],team);self.assertEqual(saved['field_side'],half)
                self.assertEqual(saved['goal_side'],'not_applicable' if label in ['set_piece','throw_in'] else 'unknown')
                self.assertEqual(result,self.store.update_segment_for_user(self.users[0],event_id,data))

    def test_throw_in_without_attribution_and_mixed_label_requirements(self):
        data=self.payload(labels=['throw_in']);data.pop('attribution_by_label')
        self.store.update_segment_for_user(self.users[0],'v0-e0',data)
        data=self.payload('v0-e1',labels=['save','shot']);data['attribution_by_label']={'save':{'field_side':'left'}}
        before=self.snapshot()
        with self.assertRaises(ValueError):self.store.update_segment_for_user(self.users[0],'v0-e1',data)
        self.assertEqual(before,self.snapshot())
        data['attribution_by_label']['shot']={'event_team':'teamB'}
        result=self.store.update_segment_for_user(self.users[0],'v0-e1',data)
        self.assertEqual({e['label'] for e in result['segment_events']},{'save','shot'})

    def test_session_reads_do_not_write_and_revocation_is_immediate(self):
        statements=[];original=self.store._open_connection
        def traced(*args,**kwargs):
            c=original(*args,**kwargs);c.set_trace_callback(statements.append);return c
        with patch.object(self.store,'_open_connection',side_effect=traced):
            for _ in range(10):self.assertEqual(self.store.session_user(self.token)['user_id'],'u0')
        self.assertFalse(any(s.lstrip().upper().startswith(('UPDATE','INSERT','DELETE')) for s in statements))
        self.store.revoke_session(self.token);self.assertIsNone(self.store.session_user(self.token))

    def test_one_commit_per_save_and_connections_closed(self):
        connections=[];statements=[];original=self.store._open_connection
        def traced(*args,**kwargs):
            c=original(*args,**kwargs);connections.append(c);c.set_trace_callback(statements.append);return c
        data=self.payload()
        with patch.object(self.store,'_open_connection',side_effect=traced):self.store.update_segment_for_user(self.users[0],'v0-e0',data)
        self.assertEqual(sum(s=='COMMIT' for s in statements),1)
        for c in connections:
            with self.assertRaises(sqlite3.ProgrammingError):c.execute('SELECT 1')

    def test_undo_monotonic_idempotent_and_stale_guarded(self):
        self.store.update_segment_for_user(self.users[0],'v0-e0',self.payload())
        data={'operation_id':secrets.token_hex(16),'expected_revision':1}
        result=self.store.undo_for_user(self.users[0],'v0-e0',data)
        self.assertEqual(result['review']['status'],'unreviewed');self.assertEqual(result['review']['revision'],2)
        self.assertEqual(result,self.store.undo_for_user(self.users[0],'v0-e0',data))
        data={**data,'operation_id':secrets.token_hex(16)}
        with self.assertRaises(v30.RevisionConflict):self.store.undo_for_user(self.users[0],'v0-e0',data)

    def test_team_save_idempotency_and_video_boundary(self):
        profile=self.store.team_profile('v0')
        data={'operation_id':secrets.token_hex(16),'expected_revision':0,'teams':profile['teams'],'status':'confirmed'}
        result=self.store.update_team_profile_for_user(self.users[0],'v0',data)
        self.assertEqual(result,self.store.update_team_profile_for_user(self.users[0],'v0',data))
        with self.assertRaises(PermissionError):self.store.update_team_profile_for_user(self.users[1],'v0',data)

    def test_redirects_complete_on_persistent_connection(self):
        c=http.client.HTTPConnection(*self.server.server_address,timeout=2)
        try:
            status,headers,body=self.request('/u/u0',connection=c)
            self.assertEqual(status,302);self.assertEqual(headers['Content-Length'],'0');self.assertEqual(body,b'')
            self.assertEqual(self.request('/healthz',connection=c)[0],200)
            c.request('POST','/auth/login',body='slug=u0&password=test-only',headers={'Content-Type':'application/x-www-form-urlencoded'})
            r=c.getresponse();self.assertEqual(r.status,302);self.assertEqual(r.read(),b'')
            self.assertEqual(self.request('/logout',connection=c)[0],302)
            self.assertEqual(self.request('/api/bootstrap',connection=c)[0],401)
        finally:c.close()

    def test_http_failed_save_does_not_commit(self):
        before=self.snapshot();data=self.payload();data['attribution_by_label']={'shot':{'field_side':'INVALID'}}
        self.assertEqual(self.request('/api/events/v0-e0/segment-decision','POST',data)[0],400)
        self.assertEqual(before,self.snapshot())

    def test_video_json_compression_preserves_data_and_respects_negotiation(self):
        import gzip
        status,headers,plain=self.request('/api/videos/v0')
        self.assertEqual(status,200)
        status,headers,compressed=self.request('/api/videos/v0',headers={'Accept-Encoding':'gzip, deflate'})
        self.assertEqual(status,200);self.assertEqual(headers['Content-Encoding'],'gzip')
        self.assertEqual(json.loads(gzip.decompress(compressed)),json.loads(plain))
        self.assertLess(len(compressed),len(plain)//2)
        _,headers,raw=self.request('/api/videos/v0',headers={'Accept-Encoding':'gzip;q=0'})
        self.assertNotIn('Content-Encoding',headers);self.assertEqual(json.loads(raw),json.loads(plain))

    def test_requested_large_ranges_are_not_artificially_truncated(self):
        media=self.root/'video.mp4'
        size=24*1024**2
        with media.open('wb') as handle:handle.truncate(size)
        for requested,expected in [('bytes=0-',f'bytes 0-{size-1}/{size}'),
                ('bytes=1024-12583935',f'bytes 1024-12583935/{size}'),
                ('bytes=-12582912',f'bytes {size-12582912}-{size-1}/{size}')]:
            status,headers,body=self.request('/media/v0','HEAD',headers={'Range':requested})
            self.assertEqual(status,206);self.assertEqual(headers['Content-Range'],expected);self.assertEqual(body,b'')
        status,headers,body=self.request('/media/v0',headers={'Range':'bytes=0-9437183'})
        self.assertEqual(status,206);self.assertEqual(len(body),9*1024**2)

    def test_ranges_cache_and_authorization(self):
        status,h,data=self.request('/media/v0',headers={'Range':'bytes=100-199'})
        self.assertEqual((status,len(data)),(206,100));self.assertEqual(data,bytes(range(100,200)))
        for value in ('bytes=-0','bytes=99999999-','bytes=2-1','bytes=1-2,4-5'):
            status,h,data=self.request('/media/v0',headers={'Range':value});self.assertEqual((status,data),(416,b''))
        self.assertEqual(self.request('/media/v1')[0],403)
        status,h,data=self.request('/media/v0',headers={'Range':'bytes=0-9'})
        self.assertEqual(self.request('/media/v0',headers={'If-None-Match':h['ETag']})[0],304)
        self.assertEqual(self.request('/media/v0',headers={'Range':'bytes=0-9','If-Range':'"old"'})[0],200)

    def test_invalid_time_is_rejected_before_writes(self):
        for value in (-1, 301, 'nan', 'inf'):
            data=self.payload();data['corrected_time_sec']=value;before=self.snapshot()
            with self.assertRaises(ValueError):self.store.update_segment_for_user(self.users[0],'v0-e0',data)
            self.assertEqual(before,self.snapshot())

    def test_concurrent_media_and_saves(self):
        payloads=[self.payload(f'v0-e{i}') for i in range(16)]
        def work(index):
            status,_,body=self.request('/media/v0',headers={'Range':'bytes=0-65535'})
            self.assertEqual((status,len(body)),(206,65536))
            status,_,body=self.request(f'/api/events/v0-e{index}/segment-decision','POST',payloads[index])
            self.assertEqual(status,200)
            return json.loads(body)['operation_id']
        with ThreadPoolExecutor(8) as pool:results=list(pool.map(work,range(16)))
        self.assertEqual(len(set(results)),16)
        self.assertEqual(sum(e['review']['revision'] for e in self.store.events_for_video('v0')),16)

    def test_head_and_request_limits_preserve_response_boundaries(self):
        c=http.client.HTTPConnection(*self.server.server_address,timeout=2)
        try:
            status,h,body=self.request('/media/v0',method='HEAD',connection=c)
            self.assertEqual((status,body),(200,b''));self.assertEqual(h['Content-Length'],'2097152')
            self.assertEqual(self.request('/healthz',connection=c)[0],200)
            status,h,body=self.request('/media/v1',method='HEAD',connection=c)
            self.assertEqual((status,body),(403,b''))
            self.assertEqual(self.request('/healthz',connection=c)[0],200)
        finally:c.close()
        status,_,_=self.request('/api/events/v0-e0/segment-decision','POST',headers={'Content-Length':'2000000'})
        self.assertEqual(status,413)

    @unittest.skipUnless(shutil.which('openssl'), 'OpenSSL required for temporary TLS fixture')
    def test_slow_tls_handshake_does_not_block_other_reviewers(self):
        certificate=self.root/'cert.pem';key=self.root/'key.pem'
        subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1',
            '-subj','/CN=localhost','-keyout',str(key),'-out',str(certificate)],check=True,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        tls=v40.ReliableHTTPServer(('127.0.0.1',0),self.server.RequestHandlerClass)
        tls.store=self.store;tls.is_tls=True
        ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);ctx.load_cert_chain(certificate,key)
        tls.socket=ctx.wrap_socket(tls.socket,server_side=True,do_handshake_on_connect=False)
        thread=threading.Thread(target=tls.serve_forever,daemon=True);thread.start()
        abandoned=socket.create_connection(tls.server_address,timeout=2)
        client=http.client.HTTPSConnection(*tls.server_address,context=ssl._create_unverified_context(),timeout=2)
        try:
            client.request('GET','/healthz');response=client.getresponse()
            self.assertEqual(response.status,200);self.assertEqual(json.loads(response.read())['version'],v40.VERSION)
        finally:
            client.close();abandoned.close();tls.shutdown();tls.server_close();thread.join(timeout=2)

    def test_effective_js_generated(self):
        text=v40.patch_js_v40((TOOL/'static/app.js').read_text())
        self.assertIn('function setupReliableQueue()',text)
        self.assertIn('generation !== videoLoadGeneration',text)
        self.assertEqual(text.count('init().catch('),1)
        self.assertIn('saveQueuePanel',v40.patch_html_v40((TOOL/'static/index.html').read_text()))

if __name__=='__main__':unittest.main()
