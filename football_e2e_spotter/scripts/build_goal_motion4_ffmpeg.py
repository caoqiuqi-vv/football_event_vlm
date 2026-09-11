#!/usr/bin/env python
"""Fast sequential-MP4 builder for 4 FPS camera-residual motion."""

from __future__ import annotations

import argparse,json,os,subprocess,time
from pathlib import Path
import cv2,numpy as np
from build_goal_motion4_bank import residual,atomic_npy


def frames(path:Path):
    command=["ffmpeg","-v","error","-i",str(path),"-vf","fps=4,scale=160:96,format=gray","-f","rawvideo","-pix_fmt","gray","pipe:1"]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE);size=160*96
    assert process.stdout is not None
    while True:
        raw=process.stdout.read(size)
        if not raw:break
        if len(raw)!=size:raise RuntimeError(f"short raw frame {len(raw)}/{size}")
        yield np.frombuffer(raw,dtype=np.uint8).reshape(96,160).astype(np.float32)/255.
    stderr=process.stderr.read().decode(errors="replace") if process.stderr else "";code=process.wait()
    if code:raise RuntimeError(f"ffmpeg failed {code}: {stderr[-1000:]}")


def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--split",required=True);p.add_argument("--shard-index",type=int,required=True);p.add_argument("--num-shards",type=int,required=True);args=p.parse_args()
    items=json.loads(args.manifest.read_text())[args.split][args.shard_index::args.num_shards]
    for item in items:
        video=str(item["media_id"]);target=args.output_root/args.split/video;meta=target/"metadata.json"
        if meta.is_file() and (target/"motion4.npy").is_file() and (target/"camera4.npy").is_file():print(json.dumps({"video_id":video,"status":"skip_ready"}),flush=True);continue
        started=time.time();motions=[];cameras=[];previous=None
        for current in frames(Path(item["source_video"])):
            motion,camera=residual(previous,current);motions.append(motion);cameras.append(camera);previous=current
        target.mkdir(parents=True,exist_ok=True);atomic_npy(target/"motion4.npy",np.asarray(motions,np.float16));atomic_npy(target/"camera4.npy",np.asarray(cameras,np.float16))
        tmp=meta.with_name(f".{meta.name}.{os.getpid()}.tmp");tmp.write_text(json.dumps({"schema":"football.goal_motion4.v1","video_id":video,"steps":len(motions),"fps":4.,"motion_dim":64,"camera_dim":3,"source":"ffmpeg_sequential"},indent=2)+"\n");os.replace(tmp,meta)
        print(json.dumps({"video_id":video,"status":"built","steps":len(motions),"seconds":round(time.time()-started,2)}),flush=True)


if __name__=="__main__":main()

