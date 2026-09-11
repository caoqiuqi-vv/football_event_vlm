"""Launch the browser regression against temporary data, never the production service."""
from pathlib import Path
import subprocess
from test_football_event_review_reliable import ReliableReviewTest

fixture = ReliableReviewTest()
fixture.setUp()
try:
    source=fixture.root/'video.mp4'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','lavfi','-i',
        'testsrc2=size=320x180:rate=15','-t','20','-c:v','libvpx-vp9','-b:v','250k','-movflags','+faststart','-y',str(source)],check=True)
    test=Path(__file__).with_name('football_review_browser_v40.cjs')
    result=subprocess.run(['node',str(test),f'http://127.0.0.1:{fixture.server.server_address[1]}'],check=False,timeout=120)
    raise SystemExit(result.returncode)
finally:
    fixture.tearDown()
