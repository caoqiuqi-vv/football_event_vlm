#!/usr/bin/env python3
"""Four-GPU sharded Verifier inference plus strict no-NMS calibration."""
from __future__ import annotations
import argparse,json,os
from collections import defaultdict
from pathlib import Path
import torch,torch.distributed as dist
from torch.utils.data import DataLoader,Dataset,DistributedSampler
from dinov3.hub.backbones import dinov3_vitl16
from football_e2e_spotter.set_data import load_set_events
from football_e2e_spotter.set_spotting import NO_EVENT_INDEX,SET_LABELS
from football_e2e_spotter.train_set_verifier import VerifierCandidateDataset,assign_oof_targets,read_rows
from football_e2e_spotter.verifier import DinoVerifier
from football_e2e_spotter.verifier_lora_runtime import configure
class Indexed(Dataset):
 def __init__(self,d):self.d=d
 def __len__(self):return len(self.d)
 def __getitem__(self,i):x=self.d[i];x['index']=torch.tensor(i);return x
def metric(rows,label,threshold,targets):
 left={k:list(v) for k,v in targets.items()};tp=fp=0
 for r in sorted((r for r in rows if r['final_label']==label and r['final_score']>=threshold),key=lambda r:r['final_score'],reverse=True):
  values=left.get(r['video_id'],[])
  if values:
   i=min(range(len(values)),key=lambda i:abs(values[i]-r['time_sec']))
   if abs(values[i]-r['time_sec'])<=3:tp+=1;values.pop(i);continue
  fp+=1
 fn=sum(len(v) for v in left.values());return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidates',required=True);p.add_argument('--annotations',required=True);p.add_argument('--verifier',required=True);p.add_argument('--output',required=True);p.add_argument('--weights',default='checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth');a=p.parse_args();dist.init_process_group('nccl');rank=dist.get_rank();local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);device=torch.device('cuda',local)
 rows=assign_oof_targets(read_rows(a.candidates),a.annotations);backbone=dinov3_vitl16(pretrained=True,weights=a.weights,check_hash=False);model=DinoVerifier(backbone,audio_dim=64).to(device);configure(backbone,warmup=False);model.load_state_dict(torch.load(a.verifier,map_location='cpu',weights_only=False)['model']);model.eval()
 ds=Indexed(VerifierCandidateDataset(rows,512,896));loader=DataLoader(ds,batch_size=1,sampler=DistributedSampler(ds,shuffle=False),num_workers=0);local_out=[]
 with torch.inference_mode():
  for b in loader:
   idx=int(b.pop('index'));b={k:v.to(device) for k,v in b.items()}
   with torch.autocast('cuda',dtype=torch.bfloat16):z=model(b['frames'],b['candidate'],b['shared'],b['audio'])
   local_out.append((idx,z['class_logits'][0].float().cpu().tolist(),float(z['quality_logits'][0].sigmoid()),float(z['time_delta_sec'][0])))
 gathered=[None]*dist.get_world_size();dist.all_gather_object(gathered,local_out)
 if rank==0:
  for part in gathered:
   for i,logit,q,delta in part:rows[i].update(verifier_logits=logit,verifier_quality=q,time_sec=rows[i]['time_sec']+delta)
  coeff=[]
  for label in range(5):
   x=torch.tensor([[r['stage1_logits'][label],r['verifier_logits'][label],r['verifier_quality']] for r in rows]);y=torch.tensor([r['target_label']==label for r in rows],dtype=torch.float);linear=torch.nn.Linear(3,1);opt=torch.optim.LBFGS(linear.parameters(),lr=.3,max_iter=100)
   def closure():opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(linear(x).squeeze(),y);loss.backward();return loss
   opt.step(closure);coeff.append({'weight':linear.weight.detach().flatten().tolist(),'bias':float(linear.bias.detach())})
   for r,s in zip(rows,linear(x).detach().squeeze().tolist()):r.setdefault('final_logits',[]).append(s)
  for r in rows:r['final_label']=int(torch.tensor(r['final_logits']).argmax());r['final_score']=float(torch.tensor(r['final_logits'])[r['final_label']].sigmoid())
  targets={i:defaultdict(list) for i in range(5)}
  for video,ann in {r['video_id']:str(r.get('annotation_id',r['video_id'])) for r in rows}.items():
   for label,t in load_set_events(Path(a.annotations)/f'{ann}.json'):targets[label][video].append(t)
  required=(.90,.85,.85,.85,.85);report={'no_temporal_nms':True,'tolerance_seconds':3.,'coefficients':coeff,'classes':{}}
  for label,name in enumerate(SET_LABELS):
   best=None
   for threshold in sorted({r['final_score'] for r in rows if r['final_label']==label},reverse=True):
    m=metric(rows,label,threshold,targets[label])
    if m['recall']>=required[label] and (best is None or m['precision']>best['precision']):best={'threshold':threshold,**m}
   report['classes'][name]=best or {'threshold':1.,**metric(rows,label,1.,targets[label])}
  output=Path(a.output);output.mkdir(parents=True,exist_ok=True);(output/'calibration_eval.json').write_text(json.dumps(report,indent=2)+'\n');(output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps(report,indent=2),flush=True)
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
