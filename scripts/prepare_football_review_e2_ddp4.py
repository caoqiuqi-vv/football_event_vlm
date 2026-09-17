#!/usr/bin/env python3
"""Validate frozen inputs and resolve local paths for the matched four-GPU E2."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
import yaml

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    repo=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-data-dir',type=Path,required=True);p.add_argument('--video-root',type=Path,required=True)
    p.add_argument('--init-checkpoint',type=Path,required=True);p.add_argument('--dino-weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu-ids',default='0,1,2,3');p.add_argument('--check-only',action='store_true');a=p.parse_args()
    gpu_ids=a.gpu_ids.split(',')
    if len(gpu_ids)!=4 or len(set(gpu_ids))!=4 or any(not x.strip() for x in gpu_ids):raise ValueError('Exactly four distinct CUDA_VISIBLE_DEVICES entries required')
    cfg=yaml.safe_load((repo/'configs/football/review_clean_e2_ddp4_20260917.yaml').read_text())
    data=a.review_data_dir.expanduser().resolve();video_root=a.video_root.expanduser().resolve();initial=a.init_checkpoint.expanduser().resolve();weights=a.dino_weights.expanduser().resolve();output=a.output.expanduser().resolve()
    manifest=data/'training_manifest.json';sha=digest(manifest)
    if sha!=cfg['data']['long_video']['review_manifest_sha256']:raise ValueError('Frozen annotation SHA256 mismatch; copy the original snapshot, do not rebuild or rewrite it')
    init_sha=digest(initial)
    if init_sha!=cfg['experiment']['init_checkpoint_sha256']:raise ValueError('Historical initialization SHA256 mismatch; do not initialize E2 from E1')
    if not weights.is_file():raise FileNotFoundError(weights)
    if not shutil.which('ffmpeg'):raise RuntimeError('ffmpeg must be installed and available on PATH')
    payload=json.loads(manifest.read_text());missing=[v['video_id'] for v in payload['videos'] if not (video_root/(v['video_id']+'.mp4')).is_file()]
    if missing:raise FileNotFoundError(f'Missing {len(missing)} original videos under {video_root}: {missing[:10]}')
    for split in ['train','val']:
        path=data/(split+'_video_ids.txt');actual=set(path.read_text().split());expected={v['video_id'] for v in payload['videos'] if v['split']==split}
        if actual!=expected:raise ValueError(f'{split} split IDs differ from frozen manifest')
    lv=cfg['data']['long_video'];lv['review_manifest']=str(manifest);lv['review_video_root']=str(video_root);lv['roots'][0]['videos_dir']=str(video_root);lv['roots'][0]['annotations_dir']=str(data);lv['split_files']={s:[str(data/(s+'_video_ids.txt'))] for s in ['train','val']}
    cfg['output_dir']=str(output);cfg['model']['init_checkpoint']=str(initial);cfg['model']['weights']=str(weights);cfg['experiment']['physical_gpu']=a.gpu_ids
    effective=4*cfg['train']['per_gpu_batch_size']*cfg['train']['grad_accum_steps'];head_lr=4*cfg['train']['lr_per_gpu'];lora_lr=4*cfg['train']['global_backbone_lr_per_gpu']
    assert effective==80 and abs(head_lr-5e-5)<1e-12 and abs(lora_lr-1e-5)<1e-12
    report={'manifest_sha256':sha,'initial_checkpoint_sha256':init_sha,'videos':len(payload['videos']),'effective_batch':effective,'head_global_lr':head_lr,'lora_global_lr':lora_lr,'frame_det_loss_weight':cfg['train']['frame_det_loss_weight'],'output':str(output),'gpu_ids':gpu_ids,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()}
    if not a.check_only:
        output.mkdir(parents=True,exist_ok=False)
        (output/'launch_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
        (output/'launch_inputs.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
