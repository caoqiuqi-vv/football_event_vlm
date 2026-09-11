"""Post-hoc calibration-only evidence sensitivity; no optimization or selection."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from football_stage2_optional import OptionalEvidence
from football_stage2_metrics import EventCurves
from football_localization_full import digest
p=ROOT/'outputs/football_localization_stage2/720p_optional_ball_roi_from_stage1e2_20260909'
out=p/'posthoc_evidence_audit';out.mkdir(exist_ok=True)
torch.set_num_threads(2);torch.manual_seed(20260909)
m=json.loads((p/'manifest.json').read_text());index=json.loads((p/'arrays/index.json').read_text());mapping={k:i for i,k in enumerate(index['keys'])}
records=m['splits']['calibration'];ids=np.array([mapping[r['key']] for r in records]);n=len(ids)
curves=EventCurves(records,m,2.)
base={}
for name in ['global_features','anchor','stage1_tokens','stage1_descriptors']:
 base[name]=torch.tensor(np.load(p/'arrays'/f'{name}.npy',mmap_mode='r')[ids],device='cuda')
base['frame_times']=torch.tensor(np.array(index['frame_times'])[ids],device='cuda',dtype=torch.float32)
report={'scope':'Post-hoc audit on calibration only, epoch6 selected in advance for diagnosis; no training, no dev/test use, no deployment claim','manifest_sha256':digest(p/'manifest.json'),'script_sha256':digest(__file__),'results':[]}
for arm,seed in [(a,s) for a in ['stage1','frozen','ordinary'] for s in [42,43]]:
 for name in ['tokens','descriptors']:
  base['stage1_'+name]=torch.tensor(np.load(p/'arrays'/f'{arm}_{name}.npy',mmap_mode='r')[ids],device='cuda')
 state=torch.load(p/f'{arm}_seed{seed}/resume.pt',map_location='cpu',weights_only=True);assert state['epoch']==6
 model=OptionalEvidence(hidden=128).cuda().eval();model.load_state_dict(state['model'])
 result={'arm':arm,'seed':seed,'epoch':6,'checkpoint_sha256':digest(p/f'{arm}_seed{seed}/resume.pt'),'modes':{}}
 predictions={}
 for mode in (['normal','cross_window','reverse_local_time','empty'] if arm=='stage1' else ['normal','empty']):
  parts=[];masses=[]
  with torch.inference_mode():
   for start in range(0,n,128):
    sl=slice(start,min(n,start+128));count=min(n-start,128)
    b={'global_features':base['global_features'][sl],'tokens':base['stage1_tokens'][sl].float(),'descriptors':base['stage1_descriptors'][sl],'anchor':base['anchor'][sl],'frame_times':base['frame_times'][sl],'valid':torch.ones(count,16,4,device='cuda',dtype=torch.bool)}
    if mode=='cross_window':
     wrong=(torch.arange(start,start+count,device='cuda')+n//2)%n
     b['tokens']=base['stage1_tokens'][wrong].float();b['descriptors']=base['stage1_descriptors'][wrong]
    if mode=='reverse_local_time':
     b['tokens']=b['tokens'].flip(1);b['descriptors']=b['descriptors'].flip(1)
    if mode=='empty':b['valid'].zero_()
    r=model(**b);parts.append(r['delta'].cpu().numpy());masses.append(r['null_mass'].cpu().numpy())
  delta=np.concatenate(parts);predictions[mode]=delta
  anchor=base['anchor'].cpu().numpy();prob=1/(1+np.exp(-np.clip(anchor+delta,-50,50)))
  aps=[]
  for c,meta in enumerate(curves.classes):
   target=np.array([bool(len(h)) for h in meta['hits']]);aps.append(float(average_precision_score(target,prob[meta['valid'],c])))
  _,metrics=curves.tune(prob,[.84,.8,.72])
  result['modes'][mode]={'mean_delta':delta.mean(0).tolist(),'std_delta':delta.std(0).tolist(),'negative_delta_fraction':(delta<0).mean(0).tolist(),'mean_null_mass':float(np.concatenate(masses).mean()),'window_AP':aps,'retuned_calibration':metrics}
 for mode in (['cross_window','reverse_local_time'] if arm=='stage1' else []):
  d=predictions[mode]-predictions['normal'];result['modes'][mode]['mean_abs_logit_change_vs_normal']=np.abs(d).mean(0).tolist()
  result['modes'][mode]['p95_abs_logit_change_vs_normal']=np.quantile(np.abs(d),.95,axis=0).tolist()
 assert np.array_equal(predictions['empty'],np.zeros_like(predictions['empty']))
 np.savez_compressed(out/f'{arm}_seed{seed}_delta.npz',**predictions)
 report['results'].append(result)
 print(json.dumps({'arm':arm,'seed':seed,'AP':result['modes']['normal']['window_AP']}),flush=True)
(out/'RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False))
