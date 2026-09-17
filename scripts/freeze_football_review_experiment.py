#!/usr/bin/env python3
"""Pin Python sources, experiment configs, initialization, and test labels."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import yaml


def main():
    repo=Path(__file__).resolve().parents[1]
    root=repo/'outputs/football_events/review_clean_e1_e2_20260916'
    source=root/'source_snapshot';source.mkdir(exist_ok=False)
    files=subprocess.run(['rg','--files','--hidden','-g','*.py','-g','!outputs/**','-g','!.git/**','-g','!.codex/**','-g','!.agents/**'],cwd=repo,capture_output=True,text=True,check=True).stdout.splitlines()
    hashes={}
    for relative in files:
        path=repo/relative;target=source/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);hashes[relative]=hashlib.sha256(target.read_bytes()).hexdigest()
    checkpoint=repo/'outputs/football_events/vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_720p_fromlast_e8_20260829/best.pt'
    anchor=root/'initial_best.pt';shutil.copy2(checkpoint,anchor)
    digest=hashlib.sha256(anchor.read_bytes()).hexdigest()
    expected=(root/'checkpoint_sha256.txt').read_text().split()[0]
    if digest!=expected:raise ValueError('Historical checkpoint changed before freeze')
    original_test=repo/'outputs/football_event_annotations/test18_final_repaired_v4_chronological_provenance_audit_20260908_per_video'
    frozen_test=root/'test18_annotations';frozen_test.mkdir()
    m=json.loads((original_test/'manifest.json').read_text());shutil.copy2(original_test/'manifest.json',frozen_test/'manifest.json')
    for item in m['files']:
        file=item.get('file',item['video_id']+'.json');shutil.copy2(original_test/file,frozen_test/file)
        if hashlib.sha256((frozen_test/file).read_bytes()).hexdigest()!=item['sha256']:raise ValueError(file)
    for stage in ('E1','E2'):
        path=repo/'configs/football'/('review_clean_'+stage.lower()+'_20260916.yaml')
        cfg=yaml.safe_load(path.read_text());cfg['model']['init_checkpoint']=str(anchor)
        cfg['experiment']['original_init_checkpoint']=str(checkpoint);cfg['experiment']['init_checkpoint_sha256']=digest
        path.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
        shutil.copy2(path,root/(stage+'.yaml'))
    provenance={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                'git_status':subprocess.check_output(['git','status','--short'],cwd=repo,text=True),
                'source_hashes':hashes,'initial_checkpoint':{'original_path':str(checkpoint),'frozen_path':str(anchor),'sha256':digest},
                'test18_source':str(original_test),'test18_frozen':str(frozen_test),
                'python':'/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python'}
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('FROZEN',len(hashes),'python files; checkpoint',digest,flush=True)


if __name__=='__main__':main()
