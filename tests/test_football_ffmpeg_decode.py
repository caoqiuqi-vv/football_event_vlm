import subprocess
import unittest
from unittest.mock import patch

from football_ffmpeg_decode import decode_timestamp_frames


class TimestampDecodeTests(unittest.TestCase):
    def result(self, pts=(0.0, 0.1), pixels=b'\x01\x02\x03\x04\x05\x06'):
        stderr='\n'.join(f'[Parsed_showinfo] n: {i} pts: {i} pts_time:{t}' for i,t in enumerate(pts)).encode()
        return subprocess.CompletedProcess([],0,pixels,stderr)

    def test_duplicate_samples_keep_pixels_and_actual_timestamps(self):
        with patch('football_ffmpeg_decode.subprocess.run',return_value=self.result()) as run:
            frames,times=decode_timestamp_frames('/source/raw.mp4',[30,30,33],30,(1,1),timeout_sec=2)
        self.assertEqual(times,[1.0,1.0,1.1])
        self.assertEqual([f.tolist() for f in frames],[[[[1,2,3]]],[[[1,2,3]]],[[[4,5,6]]]])
        self.assertEqual(run.call_args.kwargs['timeout'],2)
        self.assertIn('/source/raw.mp4',run.call_args.args[0])

    def test_native_stall_has_subprocess_timeout(self):
        with patch('football_ffmpeg_decode.subprocess.run',side_effect=subprocess.TimeoutExpired(['ffmpeg'],2)):
            with self.assertRaises(subprocess.TimeoutExpired):
                decode_timestamp_frames('/source/raw.mp4',[0],30,(1,1),timeout_sec=2)

    def test_partial_output_is_rejected(self):
        with patch('football_ffmpeg_decode.subprocess.run',return_value=self.result(pixels=b'\x01')):
            with self.assertRaisesRegex(RuntimeError,'incomplete decode'):
                decode_timestamp_frames('/source/raw.mp4',[30,33],30,(1,1))

    def test_timestamp_drift_is_rejected(self):
        with patch('football_ffmpeg_decode.subprocess.run',return_value=self.result(pts=(0.,2.))):
            with self.assertRaisesRegex(RuntimeError,'drifted'):
                decode_timestamp_frames('/source/raw.mp4',[30,33],30,(1,1))


if __name__=='__main__':unittest.main()
