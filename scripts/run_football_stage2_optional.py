#!/usr/bin/env python
"""Stage2 phases: strict frozen cache, paired controlled training, final evaluation."""
from __future__ import annotations
import argparse,json,os,sys,time,math,hashlib,subprocess,fcntl
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from football_stage2_optional import FrozenStage2Extractor,WindowFrames,OptionalEvidence
from football_stage2_metrics import EventCurves,LABELS
from football_localization_full import atomic_json,digest
OUT=ROOT/'outputs/football_localization_stage2/720p_optional_ball_roi_from_stage1e2_20260909'
PYTHON='/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python'

def save_torch(p,s):
 p=Path(p);tmp=p.with_suffix('.tmp.pt');torch.save(s,tmp);tmp.replace(p)

def unique_records(m):
 d={r['key']:r for rows in m['splits'].values() for r in rows}
 return sorted(d.values(),key=lambda r:(r['video_id'],r['start_sec'],r['key']))

def cache(cfg,m,rank):
 lock=(OUT/f'cache_rank{rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 torch.set_num_threads(1);torch.manual_seed(42);torch.cuda.set_device(0)
 sha=digest(OUT/'manifest.json');records=unique_records(m)[rank::len(cfg['gpus'])];dest=OUT/'cache';dest.mkdir(exist_ok=True)
 pending=[]
 for r in records:
  p=dest/(r['key']+'.pt')
  if p.exists():
   z=torch.load(p,weights_only=True,map_location='cpu');assert z['manifest_sha256']==sha and z['key']==r['key']
  else:pending.append(r)
 if not pending:
  atomic_json(OUT/f'cache_rank{rank}_complete.json',{'manifest_sha256':sha,'assigned':len(records),'complete':True});return
 model=FrozenStage2Extractor(cfg).cuda().eval()
 loader=DataLoader(WindowFrames(pending),batch_size=1,num_workers=cfg['workers'],pin_memory=True,prefetch_factor=1,multiprocessing_context='spawn')
 start=time.time()
 for step,(frames,index) in enumerate(loader,1):
  r=pending[int(index[0])]
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):z=model.extract_clip(frames[0].cuda(non_blocking=True),cfg['cache_frame_chunk'])
  z={k:v.detach().cpu().to(torch.float16 if k.endswith('_tokens') else torch.float32) for k,v in z.items()}
  assert all(torch.isfinite(v).all() for v in z.values());z.update(key=r['key'],manifest_sha256=sha)
  save_torch(dest/(r['key']+'.pt'),z)
  if step==1 or step%20==0:
   h={'phase':'extracting_frozen_features','rank':rank,'completed_this_run':step,'pending_at_start':len(pending),'total_assigned':len(records),'elapsed_sec':time.time()-start,'updated_unix':time.time()};atomic_json(OUT/f'cache_rank{rank}_heartbeat.json',h);print(json.dumps(h),flush=True)
 atomic_json(OUT/f'cache_rank{rank}_complete.json',{'manifest_sha256':sha,'assigned':len(records),'complete':True})

def compact(cfg,m):
 if (OUT/'COMPACT_COMPLETE.json').exists():return
 rows=unique_records(m);n=len(rows);path=OUT/'arrays';path.mkdir(exist_ok=True);sha=digest(OUT/'manifest.json')
 specs={'global_features':((n,16,2048),'float32'),'anchor':((n,3),'float32')}
 for arm in cfg['arms']:specs[arm+'_tokens']=((n,16,4,1024),'float16');specs[arm+'_descriptors']=((n,16,4,7),'float32')
 arrays={k:np.lib.format.open_memmap(path/(k+'.tmp.npy'),mode='w+',dtype=d,shape=shape) for k,(shape,d) in specs.items()}
 for i,r in enumerate(rows):
  z=torch.load(OUT/'cache'/(r['key']+'.pt'),weights_only=True,map_location='cpu');assert z['manifest_sha256']==sha and z['key']==r['key']
  for k,a in arrays.items():a[i]=z[k].numpy()
 for k,a in arrays.items():a.flush();(path/(k+'.tmp.npy')).replace(path/(k+'.npy'))
 atomic_json(path/'index.json',{'keys':[r['key'] for r in rows],'frame_times':[r['frame_times'] for r in rows],'manifest_sha256':sha})
 atomic_json(OUT/'COMPACT_COMPLETE.json',{'manifest_sha256':sha,'unique_windows':n,'all_cache_files_verified':True,'arrays':{k:{'shape':list(shape),'dtype':d,'sha256':digest(path/(k+'.npy'))} for k,(shape,d) in specs.items()}})

def train(cfg,m,arm,seed):
 torch.set_num_threads(1);torch.cuda.set_device(0);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
 dest=OUT/f'{arm}_seed{seed}';dest.mkdir(exist_ok=True);lock=(dest/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX);sha=digest(OUT/'manifest.json')
 if (dest/'COMPLETE.json').exists():return
 index=json.loads((OUT/'arrays/index.json').read_text());assert index['manifest_sha256']==sha;key_to_i={k:i for i,k in enumerate(index['keys'])};bank={}
 for name in ['global_features','anchor',arm+'_tokens',arm+'_descriptors']:
  a=np.load(OUT/'arrays'/(name+'.npy'),mmap_mode='r');bank[name]=torch.tensor(np.asarray(a),device='cuda')
 bank['frame_times']=torch.tensor(index['frame_times'],device='cuda',dtype=torch.float32)
 groups={}
 for name,rows in m['splits'].items():
  groups[name]={'indices':torch.tensor([key_to_i[r['key']] for r in rows],device='cuda'),'targets':torch.tensor([r['labels'] for r in rows],device='cuda',dtype=torch.float32),'masks':torch.tensor([r['label_mask'] for r in rows],device='cuda',dtype=torch.float32)}
 model=OptionalEvidence(hidden=cfg['hidden']).cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);curves={g:EventCurves(m['splits'][g],m,cfg['event_protocol']['tolerance_sec']) for g in ['calibration','development']}
 def batch(indices):
  return {'global_features':bank['global_features'][indices],'tokens':bank[arm+'_tokens'][indices].float(),'descriptors':bank[arm+'_descriptors'][indices],'anchor':bank['anchor'][indices],'frame_times':bank['frame_times'][indices],'valid':torch.ones(len(indices),16,4,device='cuda',dtype=torch.bool)}
 @torch.no_grad()
 def predict(group,enabled=True,stress=None):
  model.eval();ids=groups[group]['indices'];logits=[];masses=[];delta=[]
  for start in range(0,len(ids),256):
   chosen=ids[start:start+256];b=batch(chosen)
   if stress=='empty':b['valid'].zero_()
   elif stress=='half_missing':b['valid'][:,::2]=False
   elif stress=='candidate_permutation':
    wrong=batch(ids.flip(0)[start:start+len(chosen)]);b['tokens']=wrong['tokens'];b['descriptors']=wrong['descriptors']
   r=model(**b,enabled=enabled);logits.append(r['logits'].cpu());masses.append(r['null_mass'].mean(1).cpu());delta.append(r['delta'].cpu())
  logits=torch.cat(logits).numpy();return {'logits':logits,'prob':1/(1+np.exp(-np.clip(logits,-50,50))),'null_mass':torch.cat(masses).numpy(),'delta':torch.cat(delta).numpy()}
 def score(metrics):
  return float(np.mean([metrics[c]['precision'] if metrics[c]['recall']+1e-8>=floor else 0. for c,floor in zip(LABELS,cfg['recall_floors'])]))
 base=predict('calibration',False);thresholds,basecal=curves['calibration'].tune(base['prob'],cfg['recall_floors']);best={'epoch':0,'enabled':False,'score':score(basecal),'thresholds':thresholds,'calibration':basecal};begin=1
 def state(epoch):return {'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'best':best,'config':cfg,'manifest_sha256':sha,'arm':arm,'seed':seed}
 if (dest/'resume.pt').exists():
  s=torch.load(dest/'resume.pt',weights_only=True,map_location='cpu');assert s['manifest_sha256']==sha and s['config']==cfg;model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);best=s['best'];begin=s['epoch']+1
 else:save_torch(dest/'best.pt',state(0))
 train_group=groups['train'];n=len(train_group['indices']);weights=torch.tensor(cfg['pos_weight'],device='cuda')
 for epoch in range(begin,cfg['epochs']+1):
  torch.manual_seed(seed*100+epoch);order=np.random.default_rng(seed*100+epoch).permutation(n);model.train();total=0.;grad_max=0.;start=time.time()
  for step,start_i in enumerate(range(0,n,cfg['batch_size'])):
   take=torch.tensor(order[start_i:start_i+cfg['batch_size']],device='cuda');b=batch(train_group['indices'][take]);count=len(take)
   b['valid'] &= (torch.rand(count,1,1,device='cuda')>=cfg['clip_drop_probability'])
   b['valid'] &= (torch.rand(count,16,1,device='cuda')>=cfg['frame_drop_probability'])
   b['valid'] &= (torch.rand(count,16,4,device='cuda')>=cfg['candidate_drop_probability'])
   ratio=((epoch-1)*n+start_i)/(cfg['epochs']*n);opt.param_groups[0]['lr']=cfg['lr']*(.1+.9*.5*(1+math.cos(math.pi*ratio)))
   r=model(**b);mask=train_group['masks'][take];target=train_group['targets'][take]
   loss=(torch.nn.functional.binary_cross_entropy_with_logits(r['logits'],target,pos_weight=weights,reduction='none')*mask).sum()/mask.sum().clamp_min(1)
   loss=loss+cfg['residual_l2_weight']*r['delta'].square().mean()
   # Corrupted cross-window candidates are an auxiliary robustness perturbation, not absence ground truth.
   corrupted={k:v[:max(2,count//4)] for k,v in b.items()};corrupted['tokens']=corrupted['tokens'].roll(1,0);corrupted['descriptors']=corrupted['descriptors'].roll(1,0)
   loss=loss+cfg['corrupt_consistency_weight']*model(**corrupted)['delta'].square().mean()
   assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gn);opt.step();total+=float(loss.detach());grad_max=max(grad_max,float(gn))
   if step%50==0:
    h={'phase':'training_local_reader','arm':arm,'seed':seed,'epoch':epoch,'step':step+1,'steps':math.ceil(n/cfg['batch_size']),'loss':float(loss.detach()),'updated_unix':time.time()};atomic_json(dest/'heartbeat.json',h);print(json.dumps(h),flush=True)
  assert grad_max>0
  pred=predict('calibration');retuned_th,retuned=curves['calibration'].tune(pred['prob'],cfg['recall_floors']);th=thresholds;metrics=curves['calibration'].evaluate(pred['prob'],thresholds);sc=score(metrics)
  summary={'epoch':epoch,'loss_mean':total/math.ceil(n/cfg['batch_size']),'gradient_norm_max':grad_max,'calibration':metrics,'selection_score':sc,'retuned_calibration_diagnostic_only':retuned,'null_mass_mean':float(pred['null_mass'].mean()),'elapsed_sec':time.time()-start};atomic_json(dest/f'epoch_{epoch:03d}.json',summary)
  if sc>best['score']+1e-8:
   best={'epoch':epoch,'enabled':True,'score':sc,'thresholds':th,'calibration':metrics};save_torch(dest/'best.pt',state(epoch))
  save_torch(dest/'resume.pt',state(epoch))
 selected=torch.load(dest/'best.pt',weights_only=True,map_location='cpu');model.load_state_dict(selected['model']);chosen=selected['best'];dev=predict('development',chosen['enabled']);base_dev=predict('development',False);empty=predict('development',chosen['enabled'],'empty');corrupt=predict('development',chosen['enabled'],'candidate_permutation')
 assert np.array_equal(empty['logits'],base_dev['logits'])
 assert chosen['thresholds']==thresholds
 assert np.array_equal(empty['prob']>=np.asarray(thresholds),base_dev['prob']>=np.asarray(thresholds))
 partial=predict('development',chosen['enabled'],'half_missing')
 result={'arm':arm,'seed':seed,'selected':chosen,'development':curves['development'].evaluate(dev['prob'],chosen['thresholds']),'baseline_calibration':basecal,'baseline_development':curves['development'].evaluate(base_dev['prob'],thresholds),'empty_exact_anchor':True,'empty_decisions_exact_baseline':True,'common_baseline_thresholds':True,'half_missing_development':curves['development'].evaluate(partial['prob'],thresholds),'permuted_candidates_development':curves['development'].evaluate(corrupt['prob'],chosen['thresholds']),'mean_null_mass':float(dev['null_mass'].mean()),'mean_absolute_residual':float(np.abs(dev['delta']).mean()),'scope':cfg['scope'],'natural_absence_accuracy_verified':False,'manifest_sha256':sha}
 atomic_json(dest/'RESULT.json',result);np.savez_compressed(dest/'development_predictions.npz',**dev,baseline_logits=base_dev['logits'],permuted_logits=corrupt['logits'],half_missing_logits=partial['logits']);atomic_json(dest/'COMPLETE.json',{'complete':True,'epochs':cfg['epochs'],'manifest_sha256':sha})

def report(cfg,m):
 results=[]
 for arm in cfg['arms']:
  for seed in cfg['seeds']:
   dest=OUT/f'{arm}_seed{seed}';assert (dest/'COMPLETE.json').exists();r=json.loads((dest/'RESULT.json').read_text());assert r['manifest_sha256']==digest(OUT/'manifest.json');results.append(r)
 by={(r['arm'],r['seed']):r for r in results};supported=True
 for seed in cfg['seeds']:
  r=by['stage1',seed];dev=r['development'];base=r['baseline_development']
  supported &= r['selected']['enabled'] and dev['macro_precision']>base['macro_precision']
  supported &= all(dev[c]['recall']>=base[c]['recall']-.01 and dev[c]['fp_windows_per_hour']<=base[c]['fp_windows_per_hour'] for c in LABELS)
  supported &= all(dev['macro_precision']>by[a,seed]['development']['macro_precision'] for a in ['ordinary','frozen'])
  for stress in ['permuted_candidates_development','half_missing_development']:
   stressed=r[stress];supported &= all(stressed[c]['recall']>=base[c]['recall']-.01 and stressed[c]['precision']>=base[c]['precision']-.01 for c in LABELS)
 summary={'execution_complete':True,'config':cfg,'counts':m['counts'],'results':results,'stage1_increment_supported_under_prespecified_development_checks':bool(supported),'natural_object_absence_accuracy_verified':False,'independent_blind_test_claimed':False,'baseline_fallback_verified':all(r['empty_exact_anchor'] for r in results),'deployment_authorized':False}
 atomic_json(OUT/'FINAL_SUMMARY.json',summary)
 lines=['# Stage2 可选足球邻域时序实验最终报告','','两阶段数据流为 720P 原事件通路 + 冻结定位分支 + 可选择空证据的局部时序残差。无球/无球门是正常输入状态，不用检测结果筛选事件窗口。首轮仅启用足球邻域，球门不作为先决条件。','',f"训练窗口 {m['counts']['train']['windows']:,}；校准窗口 {m['counts']['calibration']['windows']:,}；完整开发滑窗 {m['counts']['development']['windows']:,}。两种子、三个匹配容量的局部分支均完成 {cfg['epochs']} 轮训练。固定帧采样和无 RGB 增强，是冻结特征条件下的受控实验。",'', '窗口精度是正窗口中覆盖已知事件的比例；事件召回按被覆盖的唯一事件计；FP/小时是错误窗口数每小时，不是去重后的事件实例误报。10秒窗口、5秒步长、边界容差2秒、无NMS。阈值仅在7个校准视频上为原模型确定一次，所有组和空证据状态共用；检查点只按校准集选择，8个开发视频用于报告。开发视频有项目历史，不称为独立盲测。','','|分支|种子|选中epoch|开发窗口宏precision|开发事件宏recall|原模型宏precision|空证据精确回退|','|---|---:|---:|---:|---:|---:|---|']
 for r in results:lines.append(f"|{r['arm']}|{r['seed']}|{r['selected']['epoch']}|{r['development']['macro_precision']*100:.2f}%|{r['development']['macro_recall']*100:.2f}%|{r['baseline_development']['macro_precision']*100:.2f}%|{r['empty_exact_anchor']}|")
 lines+=['','ordinary 为不使用检测定位的同容量局部上下文，frozen 为冻结原特征定位，stage1 为Stage1 epoch2适配定位。epoch0表示校准选型保留原模型、关闭新支路；不把无增益的已训练分支强行设为最佳。','','预先约定的开发检查结论：'+('两种子均支持 Stage1 相对基线和两个对照的增量，各类召回下降≤1pp、错误窗口/小时不增加，候选置换和半数帧缺失时各类P/R下降≤1pp。' if supported else '未同时满足两种子增量、容量对照、各类召回/误报及扰动保留约束；不能宣布 Stage2 正向通过。'),'', '所有候选不可用时，结构保证输出精确回到原事件通路。部分缺失与跨窗口错误候选参与训练，并额外评测候选置空、候选置换。置换并非真实无球标注；空证据权重是事件任务下的使用权重，不是校准后的球存在概率。没有人工存在性分层，不能宣称所有自然无球场景均已验证鲁棒。','','每类 P/R、阈值、错误窗口/小时、置换扰动结果、空证据权重、每轮校准轨迹及逐窗口预测见各分支 RESULT.json、epoch_*.json 和 development_predictions.npz。缓存来源、代码和数据清单具有指纹；原通路与定位分支全程冻结。','', '未修改或部署原事件模型。后续是否扩大训练取决于上述结果；候选检查点通过 Stage2EventModel 加载，输入仍是原始720P的16帧窗口。']
 (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n');atomic_json(OUT/'PIPELINE_COMPLETE.json',{'complete':True,'report':str(OUT/'FINAL_REPORT.md')})

def pipeline(cfg,m):
 lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 provenance=json.loads((OUT/'run_provenance.json').read_text())
 def integrity():
  assert digest(OUT/'manifest.json')==provenance['manifest_sha256']
  for p,sha in provenance['code_sha256'].items():assert digest(p)==sha,'code changed: '+p
  for k,sha in m['source_sha256'].items():assert digest(cfg[k])==sha,'source changed: '+k
 def status(phase,**kw):atomic_json(OUT/'pipeline_status.json',{'phase':phase,'supervisor_pid':os.getpid(),'updated_unix':time.time(),**kw})
 def jobs(commands,phase):
  queue=list(commands);active={};attempts={};free=list(cfg['gpus'])
  while queue or active:
   while queue and free:
    tag,args=queue.pop(0);gpu=free.pop(0);attempts[tag]=attempts.get(tag,0)+1
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    log=(OUT/(tag+'.log')).open('a');p=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),*args],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[tag]=(p,gpu,log,args)
   status(phase,active={k:{'pid':v[0].pid,'gpu':v[1],'attempt':attempts[k]} for k,v in active.items()},queued=len(queue))
   time.sleep(5)
   for tag,(p,gpu,log,args) in list(active.items()):
    code=p.poll()
    if code is None:continue
    log.close();del active[tag];free.append(gpu)
    if code:
     if attempts[tag]<2:queue.append((tag,args))
     else:
      # Preserve unrelated and still useful jobs; failure is explicit, never a success report.
      raise RuntimeError(f'{tag} failed twice with code {code}; see {tag}.log')
 try:
  integrity();status('feature_cache')
  if not (OUT/'COMPACT_COMPLETE.json').exists():
   commands=[(f'cache_rank{rank}',['--phase','cache','--rank',str(rank)]) for rank in range(len(cfg['gpus']))]
   jobs(commands,'feature_cache');integrity();status('cache_audit_and_compaction');compact(cfg,m)
  integrity();jobs([(f'{arm}_seed{seed}',['--phase','train','--arm',arm,'--seed',str(seed)]) for seed in cfg['seeds'] for arm in cfg['arms']],'reader_training_and_evaluation')
  integrity();status('final_report');report(cfg,m);status('complete',report=str(OUT/'FINAL_REPORT.md'))
 except Exception as e:
  status('failed',error=str(e));atomic_json(OUT/'PIPELINE_FAILED.json',{'error':str(e),'not_complete':True});raise

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pipeline','cache','train','report'],default='pipeline');ap.add_argument('--rank',type=int,default=0);ap.add_argument('--arm',default='stage1');ap.add_argument('--seed',type=int,default=42);args=ap.parse_args()
 cfg=json.loads((OUT/'config.json').read_text());m=json.loads((OUT/'manifest.json').read_text());assert cfg==m['config']
 if args.phase=='pipeline':pipeline(cfg,m)
 elif args.phase=='cache':cache(cfg,m,args.rank)
 elif args.phase=='train':train(cfg,m,args.arm,args.seed)
 else:report(cfg,m)
if __name__=='__main__':main()
