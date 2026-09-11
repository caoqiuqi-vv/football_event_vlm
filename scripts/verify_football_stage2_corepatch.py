"""Real 720P extraction, full factorial tiny fixture and exact restart audit."""
import copy,json,sys,shutil,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from football_stage2_corepatch import CoreExtractor,CoreEventReader,core_tokens
from football_stage2_optional import WindowFrames
from football_localization_full import atomic_json,digest
import scripts.run_football_stage2_corepatch as runner
from scripts.run_football_stage2_optional import unique_records

def main():
    torch.set_num_threads(1);torch.manual_seed(42)
    out=runner.DEFAULT;cfg=json.loads((out/'config.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    root=out/'verification';root.mkdir(exist_ok=True);fixture=root/'fixture';fixture.mkdir(exist_ok=True);base=root/'fixture_base';(base/'arrays').mkdir(parents=True,exist_ok=True)
    m=copy.deepcopy(manifest)
    for split,records in m['splits'].items():
        chosen=[]
        for c in range(3):
            found=next(r for r in records if r['labels'][c]>0 and r['label_mask'][c]>0 and r not in chosen);chosen.append(found)
        chosen.append(next(r for r in records if not any(r['labels']) and any(r['label_mask']) and r not in chosen))
        m['splits'][split]=chosen
    fc={**cfg,'output_dir':str(fixture),'base_dir':str(base),'epochs':2,'batch_size':2,'eval_batch_size':2};m['config']=fc
    atomic_json(fixture/'config.json',fc);atomic_json(fixture/'manifest.json',m);sha=digest(fixture/'manifest.json')
    records=unique_records(m);model=CoreExtractor(cfg).cuda().eval();data=WindowFrames(records);(fixture/'cache').mkdir(exist_ok=True)
    oldbase=Path(cfg['base_dir']);old_index=json.loads((oldbase/'arrays/index.json').read_text());mapping={k:i for i,k in enumerate(old_index['keys'])};ids=[mapping[r['key']] for r in records]
    for name in ['global_features','anchor','stage1_tokens','stage1_descriptors']:
        np.save(base/'arrays'/f'{name}.npy',np.load(oldbase/'arrays'/f'{name}.npy',mmap_mode='r')[ids])
    atomic_json(base/'arrays/index.json',{'keys':[r['key'] for r in records],'frame_times':[r['frame_times'] for r in records]})
    for i,r in enumerate(records):
        frames,_=data[i]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):z=model.extract_clip(frames.cuda(),8)
        old=torch.load(oldbase/'cache'/(r['key']+'.pt'),weights_only=True,map_location='cpu')
        assert torch.equal(z['global_features'].float().cpu(),old['global_features'])
        assert torch.equal(z['legacy_tokens'].half().cpu(),old['stage1_tokens'])
        assert torch.equal(z['legacy_descriptors'].float().cpu(),old['stage1_descriptors'])
        save={k:v.cpu().to(torch.float16 if k.endswith('_tokens') else torch.int16 if k.endswith('_indices') else v.dtype) for k,v in z.items() if k!='global_features' and not k.startswith('legacy')}
        save.update(key=r['key'],manifest_sha256=sha);runner.save_torch(fixture/'cache'/(r['key']+'.pt'),save)
        print(f'real 720p clip {i+1}/{len(records)} exact legacy equality',flush=True)
    del model;torch.cuda.empty_cache();runner.compact(fixture,fc,m)
    # Exact zero initialization with real 4-layer source event transformer.
    bank=runner.Bank(fixture,fc,m,'core_temporal');b=next(bank.batches(bank.groups['train'][:2],2))
    for fusion in ['temporal','late']:
        reader=CoreEventReader(fc,fusion).cuda().eval()
        with torch.no_grad():assert torch.equal(reader(**b)['logits'],b['anchor'])
        reader.train();loss=reader(**b)['logits'].square().mean();loss.backward();assert reader.adapter.output.weight.grad.abs().sum()>0
        del reader
    # Force a restart immediately AFTER epoch1 was durably committed.
    original_save=runner.save_torch
    def interrupted(path,value):
        original_save(path,value)
        if Path(path).name=='resume.pt' and value['epoch']==1:raise RuntimeError('intentional restart audit')
    runner.save_torch=interrupted
    try:runner.train(fixture,fc,m,'core_temporal',42)
    except RuntimeError as e:assert str(e)=='intentional restart audit'
    else:raise AssertionError('restart injection did not fire')
    runner.save_torch=original_save
    for arm in fc['arms']:
        for seed in fc['seeds']:runner.train(fixture,fc,m,arm,seed)
    resumed=torch.load(fixture/'core_temporal_seed42/resume.pt',weights_only=True,map_location='cpu')['adapter']
    continuous=root/'continuous';continuous.mkdir(exist_ok=True);cc={**fc,'output_dir':str(continuous)};cm=copy.deepcopy(m);cm['config']=cc
    atomic_json(continuous/'config.json',cc);atomic_json(continuous/'manifest.json',cm)
    (continuous/'arrays').symlink_to(fixture/'arrays',target_is_directory=True)
    runner.train(continuous,cc,cm,'core_temporal',42)
    actual=torch.load(continuous/'core_temporal_seed42/resume.pt',weights_only=True,map_location='cpu')['adapter']
    assert all(torch.equal(v,actual[k]) for k,v in resumed.items()),'restart differs from uninterrupted training'
    runner.report(fixture,fc,m)
    # Completed jobs must not mutate any checkpoint on resume.
    checkpoint=fixture/'core_temporal_seed42/resume.pt';before=digest(checkpoint);runner.train(fixture,fc,m,'core_temporal',42);assert digest(checkpoint)==before
    atomic_json(out/'VERIFICATION_COMPLETE.json',{'real_720p_clips':len(records),'old_global_roi_and_descriptors_exact':True,'real_event_zero_initialization_exact':True,'real_event_adapter_gradient_verified':True,'factorial_jobs':12,'fixture_epochs':2,'interrupted_resume_tensor_exact':True,'completed_resume_immutable':True,'not_scientific_efficacy':True,'updated_unix':time.time()})
    print('VERIFICATION COMPLETE',flush=True)
if __name__=='__main__':main()
