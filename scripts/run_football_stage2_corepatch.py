#!/usr/bin/env python
"""Resumable factorial experiment on exact core patches and fusion placement."""
from __future__ import annotations
import argparse,json,os,sys,time,math,subprocess,fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from football_stage2_corepatch import CoreExtractor,CoreEventReader,legacy_descriptors
from football_stage2_optional import WindowFrames
from football_stage2_metrics import EventCurves,LABELS
from football_localization_full import atomic_json,digest
from scripts.run_football_stage2_optional import unique_records,save_torch
PYTHON='/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python'
DEFAULT=ROOT/'outputs/football_localization_stage2/720p_corepatch_crossattn_factorial_from_stage1e2_20260909'


def cache(out,cfg,m,rank):
    lock=(out/f'cache_rank{rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    sha=digest(out/'manifest.json');old_sha=digest(Path(cfg['base_dir'])/'manifest.json');done=out/f'cache_rank{rank}_complete.json'
    if done.exists():assert json.loads(done.read_text())['manifest_sha256']==sha;return
    torch.set_num_threads(1);torch.manual_seed(42)
    records=unique_records(m)[rank::len(cfg['gpus'])];dest=out/'cache';dest.mkdir(exist_ok=True)
    pending=[]
    for r in records:
        p=dest/(r['key']+'.pt')
        if p.exists():
            z=torch.load(p,weights_only=True,map_location='cpu');assert z['manifest_sha256']==sha and z['key']==r['key']
        else:pending.append(r)
    model=CoreExtractor(cfg).cuda().eval();start=time.time()
    loader=DataLoader(WindowFrames(pending),batch_size=1,num_workers=cfg['workers'],pin_memory=True,prefetch_factor=1,multiprocessing_context='spawn')
    for step,(frames,index) in enumerate(loader,1):
        record=pending[int(index[0])]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):z=model.extract_clip(frames[0].cuda(non_blocking=True),8)
        old=torch.load(Path(cfg['base_dir'])/'cache'/(record['key']+'.pt'),weights_only=True,map_location='cpu')
        assert old['key']==record['key'] and old['manifest_sha256']==old_sha
        assert torch.equal(z['global_features'].float().cpu(),old['global_features']),record['key']+': original frame feature mismatch'
        assert torch.equal(z['legacy_tokens'].half().cpu(),old['stage1_tokens']),record['key']+': legacy ROI mismatch'
        assert torch.equal(z['legacy_descriptors'].float().cpu(),old['stage1_descriptors']),record['key']+': heatmap selection mismatch'
        result={}
        for name,v in z.items():
            if name=='global_features' or name.startswith('legacy'):continue
            dtype=torch.float16 if name.endswith('_tokens') else torch.int16 if name.endswith('_indices') else v.dtype
            result[name]=v.detach().cpu().to(dtype)
            assert torch.isfinite(result[name]).all()
        result.update(manifest_sha256=sha,key=record['key']);save_torch(dest/(record['key']+'.pt'),result)
        if step==1 or step%20==0:
            h={'phase':'720p_core_patch_extraction','rank':rank,'completed_this_run':step,'pending_at_start':len(pending),'total_assigned':len(records),'elapsed_sec':time.time()-start,'updated_unix':time.time()};atomic_json(out/f'cache_rank{rank}_heartbeat.json',h);print(json.dumps(h),flush=True)
    atomic_json(done,{'manifest_sha256':sha,'assigned':len(records),'complete':True,'old_feature_and_roi_exact':True})


def compact(out,cfg,m):
    if (out/'COMPACT_COMPLETE.json').exists():return
    records=unique_records(m);n=len(records);sha=digest(out/'manifest.json');path=out/'arrays';path.mkdir(exist_ok=True)
    specs={}
    for arm in ['stage1','frozen','ordinary']:
        for name,tail,dtype in [('tokens',(16,26,1024),'float16'),('descriptors',(16,26,11),'float32'),('valid',(16,26),'bool'),('patch_indices',(16,26),'int16')]:specs[arm+'_'+name]=((n,*tail),dtype)
    arrays={k:np.lib.format.open_memmap(path/(k+'.tmp.npy'),mode='w+',dtype=d,shape=shape) for k,(shape,d) in specs.items()}
    for i,r in enumerate(records):
        z=torch.load(out/'cache'/(r['key']+'.pt'),weights_only=True,map_location='cpu');assert z['key']==r['key'] and z['manifest_sha256']==sha
        for name,a in arrays.items():a[i]=z[name].numpy()
        if i%1000==0:atomic_json(out/'compaction_heartbeat.json',{'completed':i,'total':n,'updated_unix':time.time()})
    for name,a in arrays.items():a.flush();(path/(name+'.tmp.npy')).replace(path/(name+'.npy'))
    oldindex=json.loads((Path(cfg['base_dir'])/'arrays/index.json').read_text());assert oldindex['keys']==[r['key'] for r in records]
    atomic_json(path/'index.json',{'keys':oldindex['keys'],'frame_times':oldindex['frame_times'],'manifest_sha256':sha})
    atomic_json(out/'COMPACT_COMPLETE.json',{'manifest_sha256':sha,'windows':n,'arrays':{k:{'shape':list(shape),'dtype':d,'sha256':digest(path/(k+'.npy'))} for k,(shape,d) in specs.items()},'exact_old_features_and_rois_verified':True})


class Bank:
    def __init__(self,out,cfg,m,arm):
        self.arm=cfg['arms'][arm];old=Path(cfg['base_dir'])/'arrays';index=json.loads((old/'index.json').read_text())
        mapping={k:i for i,k in enumerate(index['keys'])};self.groups={s:np.array([mapping[r['key']] for r in rows]) for s,rows in m['splits'].items()}
        self.times=np.array(index['frame_times'],np.float32)
        self.arrays={k:np.load(old/(k+'.npy'),mmap_mode='r') for k in ['global_features','anchor']}
        name=self.arm['source']
        path=old if self.arm['representation']=='roi' else out/'arrays'
        for k in ['tokens','descriptors']:self.arrays[k]=np.load(path/f'{name}_{k}.npy',mmap_mode='r')
        self.valid=None if self.arm['representation']=='roi' else np.load(path/f'{name}_valid.npy',mmap_mode='r')
    def cpu(self,indices):
        b={k:torch.from_numpy(np.array(a[indices],copy=True)) for k,a in self.arrays.items()}
        if self.arm['representation']=='roi':b['descriptors']=legacy_descriptors(b['descriptors'])
        b['valid']=torch.ones(b['tokens'].shape[:-1],dtype=torch.bool) if self.valid is None else torch.from_numpy(np.array(self.valid[indices],copy=True))
        b['frame_times']=torch.from_numpy(self.times[indices].copy())
        return {k:v.pin_memory() for k,v in b.items()}
    def batches(self,indices,size):
        slices=[indices[i:i+size] for i in range(0,len(indices),size)]
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending=pool.submit(self.cpu,slices[0]) if slices else None
            for i in range(len(slices)):
                b=pending.result()
                if i+1<len(slices):pending=pool.submit(self.cpu,slices[i+1])
                yield {k:v.cuda(non_blocking=True) for k,v in b.items()}


def window_ap(curves,prob):
    per=[]
    for c,meta in enumerate(curves.classes):
        y=np.array([bool(len(x)) for x in meta['hits']]);per.append(float(average_precision_score(y,prob[meta['valid'],c])) if y.any() else 0.)
    return {'per_class':dict(zip(LABELS,per)),'macro':float(np.mean(per))}


def train(out,cfg,m,arm,seed):
    dest=out/f'{arm}_seed{seed}';dest.mkdir(exist_ok=True);lock=(dest/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if (dest/'COMPLETE.json').exists():return
    torch.set_num_threads(1);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    bank=Bank(out,cfg,m,arm);model=CoreEventReader(cfg,cfg['arms'][arm]['fusion']).cuda()
    params=list(model.adapter.parameters());opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    sha=digest(out/'manifest.json');curves={s:EventCurves(m['splits'][s],m,2.) for s in ['calibration','development']}
    @torch.no_grad()
    def predict(group,enabled=True,stress=None):
        model.eval();parts=[];null=[];deltas=[];indices=bank.groups[group]
        for i,b in enumerate(bank.batches(indices,cfg['eval_batch_size'])):
            if stress=='empty':b['valid'].zero_()
            elif stress=='half_missing':b['valid'][:,::2]=False
            elif stress=='reverse_local_time':
                for k in ['tokens','descriptors','valid']:b[k]=b[k].flip(1)
            elif stress=='cross_window':
                start=i*cfg['eval_batch_size'];wrong=indices[(np.arange(start,start+len(b['tokens']))+len(indices)//2)%len(indices)]
                corrupt={k:v.cuda(non_blocking=True) for k,v in bank.cpu(wrong).items()}
                for k in ['tokens','descriptors','valid']:b[k]=corrupt[k]
            r=model(**b,enabled=enabled);parts.append(r['logits'].cpu().numpy());null.append(r['null_mass'].cpu().numpy());deltas.append(r['delta'].cpu().numpy())
        logits=np.concatenate(parts);return {'logits':logits,'prob':1/(1+np.exp(-np.clip(logits,-50,50))),'delta':np.concatenate(deltas),'null_mass':np.concatenate(null)}
    basecal=predict('calibration',False);base_th,base_metrics=curves['calibration'].tune(basecal['prob'],cfg['recall_floors'])
    baseap=window_ap(curves['calibration'],basecal['prob']);best={'epoch':0,'enabled':False,'score':baseap['macro'],'thresholds':base_th,'calibration':base_metrics};begin=1
    def state(epoch):return {'epoch':epoch,'adapter':model.adapter.state_dict(),'optimizer':opt.state_dict(),'best':best,'baseline_thresholds':base_th,'config':cfg,'manifest_sha256':sha,'arm':arm,'seed':seed}
    if (dest/'resume.pt').exists():
        s=torch.load(dest/'resume.pt',weights_only=True,map_location='cpu');assert s['manifest_sha256']==sha and s['config']==cfg
        model.adapter.load_state_dict(s['adapter']);opt.load_state_dict(s['optimizer']);best=s['best'];begin=s['epoch']+1
    else:
        initial=predict('calibration',True);assert np.array_equal(initial['logits'],basecal['logits']),'zero initialization not exact'
        save_torch(dest/'best.pt',state(0))
    rows=m['splits']['train'];targets=torch.tensor([r['labels'] for r in rows],device='cuda',dtype=torch.float32);masks=torch.tensor([r['label_mask'] for r in rows],device='cuda',dtype=torch.float32);n=len(rows)
    weights=torch.tensor(cfg['pos_weight'],device='cuda');start_all=time.time()
    for epoch in range(begin,cfg['epochs']+1):
        torch.manual_seed(seed*100+epoch);order=np.random.default_rng(seed*100+epoch).permutation(n);model.train();losses=[];grad_max=0.;start=time.time()
        for step,b in enumerate(bank.batches(bank.groups['train'][order],cfg['batch_size'])):
            take=torch.as_tensor(order[step*cfg['batch_size']:(step+1)*cfg['batch_size']],device='cuda');count=len(take)
            keep_clip=torch.rand(count,1,1,device='cuda')>=cfg['clip_drop_probability'];keep_frame=torch.rand(count,16,1,device='cuda')>=cfg['frame_drop_probability']
            keep_candidate=torch.rand(count,16,2,device='cuda')>=cfg['candidate_drop_probability'];which=b['descriptors'][...,10].long().clamp(0,1)
            b['valid'] &= keep_clip & keep_frame & keep_candidate.gather(2,which)
            ratio=((epoch-1)*n+step*cfg['batch_size'])/(cfg['epochs']*n);opt.param_groups[0]['lr']=cfg['lr']*(.1+.9*.5*(1+math.cos(math.pi*ratio)))
            r=model(**b);mask=masks[take];loss=(torch.nn.functional.binary_cross_entropy_with_logits(r['logits'],targets[take],pos_weight=weights,reduction='none')*mask).sum()/mask.sum().clamp_min(1)
            loss=loss+cfg['residual_l2_weight']*r['delta'].square().mean()
            corrupt={k:v[:max(2,count//4)] for k,v in b.items()}
            for k in ['tokens','descriptors','valid']:corrupt[k]=corrupt[k].roll(1,0)
            loss=loss+cfg['corrupt_consistency_weight']*model(**corrupt)['delta'].square().mean()
            assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(gn);opt.step()
            losses.append(float(loss.detach()));grad_max=max(grad_max,float(gn))
            if step%25==0:
                h={'phase':'training','arm':arm,'seed':seed,'epoch':epoch,'step':step+1,'steps':math.ceil(n/cfg['batch_size']),'loss':float(loss.detach()),'elapsed_sec':time.time()-start,'updated_unix':time.time()};atomic_json(dest/'heartbeat.json',h);print(json.dumps(h),flush=True)
        assert grad_max>0
        pred=predict('calibration');ap=window_ap(curves['calibration'],pred['prob']);thresholds,metrics=curves['calibration'].tune(pred['prob'],cfg['recall_floors'])
        fixed=curves['calibration'].evaluate(pred['prob'],base_th)
        if ap['macro']>best['score']+1e-8:
            best={'epoch':epoch,'enabled':True,'score':ap['macro'],'thresholds':thresholds,'calibration':metrics};save_torch(dest/'best.pt',state(epoch))
        summary={'epoch':epoch,'loss_mean':float(np.mean(losses)),'gradient_norm_max':grad_max,'calibration_window_AP':ap,'calibration_retuned':metrics,'calibration_fixed_thresholds':fixed,'selection':'calibration macro window AP','best_epoch':best['epoch'],'null_mass_mean':float(pred['null_mass'].mean()),'elapsed_sec':time.time()-start}
        atomic_json(dest/f'epoch_{epoch:03d}.json',summary);save_torch(dest/f'epoch_{epoch:03d}.pt',state(epoch));save_torch(dest/'resume.pt',state(epoch))
    # Audit the trained epoch6 with the branch ON regardless of model selection.
    last=predict('development',True);last_stress={}
    for stress in ['empty','half_missing','cross_window','reverse_local_time']:
        pred=predict('development',True,stress)
        if stress=='empty':assert np.array_equal(pred['logits'],np.asarray(bank.arrays['anchor'][bank.groups['development']]))
        last_stress[stress]={'window_AP':window_ap(curves['development'],pred['prob']),'fixed_thresholds':curves['development'].evaluate(pred['prob'],base_th),'mean_absolute_residual':float(np.abs(pred['delta']).mean())}
        np.savez_compressed(dest/f'last_enabled_{stress}.npz',logits=pred['logits'])
    np.savez_compressed(dest/'last_enabled.npz',logits=last['logits'])
    chosen=torch.load(dest/'best.pt',weights_only=True,map_location='cpu')['best'];model.adapter.load_state_dict(torch.load(dest/'best.pt',weights_only=True,map_location='cpu')['adapter'])
    dev=predict('development',chosen['enabled']);baseline=predict('development',False);empty=predict('development',True,'empty')
    assert np.array_equal(empty['logits'],baseline['logits'])
    selected_stress={}
    for stress in ['empty','half_missing','cross_window','reverse_local_time']:
        pred=predict('development',True,stress)
        thresholds=base_th if stress=='empty' else chosen['thresholds']
        selected_stress[stress]={'window_AP':window_ap(curves['development'],pred['prob']),'operating_metrics':curves['development'].evaluate(pred['prob'],thresholds),'mean_absolute_residual':float(np.abs(pred['delta']).mean())}
        np.savez_compressed(dest/f'selected_enabled_{stress}.npz',logits=pred['logits'])
    result={'arm':arm,'seed':seed,'selected_enabled_stress':selected_stress,'selected_stress_note':'adapter forced enabled even if selection chose epoch0; epoch0 adapter is zero initialized','selected':chosen,'baseline_thresholds':base_th,'baseline_calibration_AP':baseap,'development_AP':window_ap(curves['development'],dev['prob']),'baseline_development_AP':window_ap(curves['development'],baseline['prob']),'development_retuned':curves['development'].evaluate(dev['prob'],chosen['thresholds']),'development_fixed':curves['development'].evaluate(dev['prob'],base_th),'baseline_development':curves['development'].evaluate(baseline['prob'],base_th),'last_enabled_AP':window_ap(curves['development'],last['prob']),'last_enabled_stress':last_stress,'explicit_empty_logits_exact':True,'empty_decision_rule':'use baseline thresholds when all candidates explicitly invalid; normal inputs use selected thresholds','natural_absence_verified':False,'trainable_parameters':sum(p.numel() for p in params),'manifest_sha256':sha,'elapsed_sec':time.time()-start_all}
    np.savez_compressed(dest/'development_predictions.npz',logits=dev['logits'],baseline_logits=baseline['logits'])
    atomic_json(dest/'RESULT.json',result);atomic_json(dest/'COMPLETE.json',{'complete':True,'epochs':cfg['epochs'],'manifest_sha256':sha})


def paired_video_bootstrap(records,curves,left,right,repeats=500):
    videos=sorted({r['video_id'] for r in records});vi={v:i for i,v in enumerate(videos)}
    rng=np.random.default_rng(20260909);deltas=[]
    lp=1/(1+np.exp(-np.clip(left,-50,50)));rp=1/(1+np.exp(-np.clip(right,-50,50)))
    metas=[]
    for c,meta in enumerate(curves.classes):
        y=np.array([bool(len(h)) for h in meta['hits']]);ids=np.array([vi[records[j]['video_id']] for j in meta['valid']]);metas.append((y,ids,meta['valid']))
    for _ in range(repeats):
        counts=np.bincount(rng.integers(0,len(videos),len(videos)),minlength=len(videos));scores=[]
        for c,(y,ids,valid) in enumerate(metas):
            weight=counts[ids]
            if not weight[y].sum():break
            scores.append(average_precision_score(y,lp[valid,c],sample_weight=weight)-average_precision_score(y,rp[valid,c],sample_weight=weight))
        if len(scores)==3:deltas.append(float(np.mean(scores)*100))
    return {'metric':'macro window AP delta pp','videos':len(videos),'bootstrap_repeats':repeats,'valid_replicates':len(deltas),'percentile_95CI_pp':np.quantile(deltas,[.025,.975]).tolist() if deltas else None,'interpretation':'paired video resampling on historically used development videos; not independent test evidence'}


def report(out,cfg,m):
    rows=[]
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            d=out/f'{arm}_seed{seed}';assert (d/'COMPLETE.json').exists();r=json.loads((d/'RESULT.json').read_text());assert r['manifest_sha256']==digest(out/'manifest.json');rows.append(r)
    by={(r['arm'],r['seed']):r for r in rows};positive=True
    for seed in cfg['seeds']:
        r=by['core_temporal',seed];dev=r['development_retuned'];base=r['baseline_development']
        positive &= r['selected']['enabled'] and r['development_AP']['macro']>r['baseline_development_AP']['macro'] and dev['macro_precision']>base['macro_precision']
        for other in ['roi_temporal','core_late','ordinary_core_temporal','frozen_core_temporal']:
            positive &= r['development_AP']['macro']>by[other,seed]['development_AP']['macro']
        positive &= all(dev[c]['recall']>=base[c]['recall']-.01 and dev[c]['fp_windows_per_hour']<=base[c]['fp_windows_per_hour'] for c in LABELS)
    bootstrap={}
    curves=EventCurves(m['splits']['development'],m,2.)
    for seed in cfg['seeds']:
        current=np.load(out/f'core_temporal_seed{seed}/development_predictions.npz')
        for other in ['baseline','roi_temporal','core_late','ordinary_core_temporal','frozen_core_temporal']:
            reference=current['baseline_logits'] if other=='baseline' else np.load(out/f'{other}_seed{seed}/development_predictions.npz')['logits']
            bootstrap[f'seed{seed}_vs_{other}']=paired_video_bootstrap(m['splits']['development'],curves,current['logits'],reference)
    uncertainty_supported=all(v['percentile_95CI_pp'] is not None and v['percentile_95CI_pp'][0]>0 for v in bootstrap.values())
    robust=True
    for seed in cfg['seeds']:
        selected=by['core_temporal',seed];base=selected['baseline_development']
        robust &= selected['selected']['enabled']
        for stress in ['half_missing','cross_window']:
            metrics=selected.get('selected_enabled_stress',{}).get(stress,{}).get('operating_metrics')
            robust &= metrics is not None and all(metrics[c]['precision']>=base[c]['precision']-.01 and metrics[c]['recall']>=base[c]['recall']-.01 for c in LABELS)
    summary={'paired_video_uncertainty_supported':bool(uncertainty_supported),'synthetic_robustness_checks_passed':bool(robust),'overall_experimental_checks_passed':bool(positive and uncertainty_supported and robust),'paired_video_bootstrap':bootstrap,'execution_complete':True,'results':rows,'prespecified_normal_input_checks_passed':bool(positive),'natural_absence_verified':False,'independent_test_claimed':False,'scope':'historically used calibration/development video split; normal-input efficacy and synthetic robustness reported separately'}
    atomic_json(out/'FINAL_SUMMARY.json',summary)
    lines=['# Stage2 核心 patch 与时序交叉注意力实验','', '720P、16帧、10秒窗口；核心中心 patch 原样保留，两个候选及周边网格，跨相邻帧软注意力和 NULL。原事件模型与 Stage1 冻结，仅训练同容量适配器。','', '六组、两个种子均完成6轮。检查点按校准窗口宏 AP 选择；阈值仅在校准集按召回下限选择。原阈值结果另报。AP 是窗口排序 AP，不是事件 spotting mAP。开发视频有历史使用，不是盲测。','', '|分支|种子|最佳epoch|开发宏窗口AP|原模型AP|开发窗口P|开发事件R|','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['arm']}|{r['seed']}|{r['selected']['epoch']}|{r['development_AP']['macro']*100:.3f}%|{r['baseline_development_AP']['macro']*100:.3f}%|{r['development_retuned']['macro_precision']*100:.3f}%|{r['development_retuned']['macro_recall']*100:.3f}%|")
    lines+=['','正常输入预注册检查：'+('通过；仍需审查合成扰动及未使用测试集，不能宣布完整鲁棒部署通过。' if positive else '未通过，不能宣布 Stage1 信息已可靠转化为事件增量。'),'','每组 RESULT.json 单独报告实际启用 epoch6 的空输入、部分缺失、跨窗错配和局部时间反转；这些不是自然无球标注。所有候选显式缺失时恢复原模型 logits，并使用原阈值；正常输入使用该分支校准阈值，因此不能称为普遍兼容原阈值。','', 'roi_late/core_late 对比空间保留；roi_temporal/core_temporal 为同一因素的交叉验证；core_late/core_temporal 对比分数读出与原事件时序层融合；普通区域和冻结定位分支用于检验 Stage1 增量。此处 late 为在冻结原分类头前加入证据、转换为 logit 残差，同容量适配器重新训练；旧实验原分支结果仅作历史参考。','', '完整代码、配置、源模型和数据指纹见 run_provenance.json。结果不修改或部署原事件模型。']
    lines+=['',f'视频重采样区间支持增量：{bool(uncertainty_supported)}；所选模型启用时合成缺失/错配检查通过：{bool(robust)}。不能替代自然无球分层或独立测试。','','配对视频bootstrap（宏窗口AP差，百分点；95%区间）：']
    for name,ci in bootstrap.items():lines.append(f"- {name}: {ci['percentile_95CI_pp']}")
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines));atomic_json(out/'PIPELINE_COMPLETE.json',{'complete':True,'report':str(out/'FINAL_REPORT.md')})


def pipeline(out,cfg,m):
    lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    provenance=json.loads((out/'run_provenance.json').read_text())
    def integrity():
        for path,sha in provenance['sha256'].items():assert digest(path)==sha,'provenance changed: '+path
    def status(phase,**kw):atomic_json(out/'pipeline_status.json',{'phase':phase,'supervisor_pid':os.getpid(),'updated_unix':time.time(),**kw})
    def jobs(tasks,phase):
        queue=list(tasks);active={};free=list(cfg['gpus']);attempts={};errors=[]
        while queue or active:
            gpu_text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            available={int(line.split(',')[0]) for line in gpu_text.strip().splitlines() if int(line.split(',')[1])<256}
            ready=[g for g in free if g in available]
            while queue and ready and not errors:
                tag,args=queue.pop(0);gpu=ready.pop(0);free.remove(gpu);attempts[tag]=attempts.get(tag,0)+1
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                log=(out/(tag+'.log')).open('a');proc=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'--output-dir',str(out),*args],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[tag]=(proc,gpu,log,args)
            status(phase,active={k:{'pid':v[0].pid,'gpu':v[1],'attempt':attempts[k]} for k,v in active.items()},queued=len(queue),waiting_gpus=[g for g in free if g not in available],errors=errors)
            time.sleep(5)
            for tag,(proc,gpu,log,args) in list(active.items()):
                code=proc.poll()
                if code is None:continue
                log.close();del active[tag];free.append(gpu)
                if code:
                    if attempts[tag]<2:queue.append((tag,args))
                    else:errors.append(f'{tag}: exit {code}')
            if errors and not active:raise RuntimeError('; '.join(errors))
    try:
        integrity()
        if not (out/'COMPACT_COMPLETE.json').exists():
            jobs([(f'cache_rank{i}',['--phase','cache','--rank',str(i)]) for i in range(len(cfg['gpus']))],'feature_cache')
            integrity();status('compacting');compact(out,cfg,m)
        integrity();jobs([(f'{arm}_seed{seed}',['--phase','train','--arm',arm,'--seed',str(seed)]) for seed in cfg['seeds'] for arm in cfg['arms']],'training_and_evaluation')
        integrity();status('final_report');report(out,cfg,m);status('complete')
    except Exception as e:
        status('failed',error=str(e));atomic_json(out/'PIPELINE_FAILED.json',{'error':str(e),'not_complete':True});raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=DEFAULT);ap.add_argument('--phase',choices=['pipeline','cache','compact','train','report'],default='pipeline');ap.add_argument('--rank',type=int,default=0);ap.add_argument('--arm',default='core_temporal');ap.add_argument('--seed',type=int,default=42);args=ap.parse_args()
    out=args.output_dir.resolve();cfg=json.loads((out/'config.json').read_text());m=json.loads((out/'manifest.json').read_text());assert m['config']==cfg
    if args.phase=='pipeline':pipeline(out,cfg,m)
    elif args.phase=='cache':cache(out,cfg,m,args.rank)
    elif args.phase=='compact':compact(out,cfg,m)
    elif args.phase=='train':train(out,cfg,m,args.arm,args.seed)
    else:report(out,cfg,m)
if __name__=='__main__':main()
