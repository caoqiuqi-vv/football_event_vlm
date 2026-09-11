#!/usr/bin/env python
"""Build detector-free 4 FPS camera-residual motion timelines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import lmdb
import numpy as np


def read_gray(txn, index: int) -> np.ndarray:
    value=txn.get(f"f/{index:08d}".encode())
    if value is None: raise KeyError(index)
    gray=cv2.imdecode(np.frombuffer(value,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    if gray is None: raise RuntimeError(f"corrupt frame {index}")
    return cv2.resize(gray,(160,96),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0


def residual(previous: np.ndarray|None,current: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    if previous is None: return np.zeros(64,np.float32),np.zeros(3,np.float32)
    shift,response=cv2.phaseCorrelate(previous,current)
    stable=np.isfinite(shift).all() and response>=.02 and abs(shift[0])<=48 and abs(shift[1])<=29
    if stable:
        transform=np.asarray([[1,0,shift[0]],[0,1,shift[1]]],np.float32)
        aligned=cv2.warpAffine(previous,transform,(160,96),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    else: aligned=previous; shift=(0.,0.); response=0.
    grid=cv2.resize(cv2.absdiff(current,aligned),(8,8),interpolation=cv2.INTER_AREA).reshape(-1)
    camera=np.asarray([shift[0]/160.,shift[1]/96.,response],np.float32)
    return grid.astype(np.float32),camera


def atomic_npy(path:Path,array:np.ndarray):
    tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as handle: np.save(handle,array,allow_pickle=False)
    os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--pixel-root",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--split",required=True);p.add_argument("--shard-index",type=int,required=True);p.add_argument("--num-shards",type=int,required=True);args=p.parse_args()
    items=json.loads(args.manifest.read_text(encoding="utf-8"))[args.split][args.shard_index::args.num_shards]
    for item in items:
        video=str(item["media_id"]); source=args.pixel_root/args.split/video; target=args.output_root/args.split/video; meta=target/"metadata.json"
        if meta.is_file() and (target/"motion4.npy").is_file() and (target/"camera4.npy").is_file(): print(json.dumps({"video_id":video,"status":"skip_ready"}),flush=True);continue
        source_meta=json.loads((source/"metadata.json").read_text()); count=int(source_meta["frame_count"])
        env=lmdb.open(str(source/"frames.lmdb"),readonly=True,lock=False,readahead=False,max_readers=8); motions=[];cameras=[];previous=None
        with env.begin(buffers=True) as txn:
            for index in range(count):
                current=read_gray(txn,index); motion,camera=residual(previous,current);motions.append(motion);cameras.append(camera);previous=current
        env.close();target.mkdir(parents=True,exist_ok=True);atomic_npy(target/"motion4.npy",np.asarray(motions,np.float16));atomic_npy(target/"camera4.npy",np.asarray(cameras,np.float16))
        tmp=meta.with_name(f".{meta.name}.{os.getpid()}.tmp");tmp.write_text(json.dumps({"schema":"football.goal_motion4.v1","video_id":video,"steps":count,"fps":4.0,"motion_dim":64,"camera_dim":3},indent=2)+"\n");os.replace(tmp,meta)
        print(json.dumps({"video_id":video,"status":"built","steps":count}),flush=True)


if __name__=="__main__":main()

