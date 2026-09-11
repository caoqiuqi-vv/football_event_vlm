#!/usr/bin/env python
"""Priority-ordered Stage2 joint temporal adaptation; reuses immutable core cache."""
import argparse,json,os,sys,time,subprocess,fcntl
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from football_stage2_metrics import EventCurves,LABELS
from football_localization_full import atomic_json,digest
from scripts.run_football_stage2_optional import save_torch
from football_events.stage2.data import Bank
from football_events.stage2.training import train
from football_events.stage2.metrics import window_ap, paired_video_bootstrap
DEFAULT=ROOT/'outputs/football_localization_stage2/720p_corepatch_joint_temporal_priority_20260909'
PYTHON='/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python'





def report(out,cfg,m):
    results=[]
    for arm,seed in cfg['execution_order']:
        p=out/f'{arm}_seed{seed}';assert (p/'COMPLETE.json').exists()
        r=json.loads((p/'RESULT.json').read_text());assert r['manifest_sha256']==digest(out/'manifest.json');results.append(r)
    by={(r['arm'],r['seed']):r for r in results};curves=EventCurves(m['splits']['development'],m,2.)
    intervals={};point=True;robust=True
    for seed in cfg['seeds']:
        a=by['joint_temporal',seed];f=by['frozen_temporal',seed];base=a['baseline_development'];dev=a['development_retuned']
        point &= a['selected']['enabled'] and a['development_AP']['macro']>a['baseline_development_AP']['macro'] and a['development_AP']['macro']>f['development_AP']['macro']
        point &= dev['macro_precision']>base['macro_precision'] and all(dev[c]['recall']>=base[c]['recall']-.01 and dev[c]['fp_windows_per_hour']<=base[c]['fp_windows_per_hour'] for c in LABELS)
        for stress in ['half_missing','cross_window']:
            metric=a['selected_enabled_stress'][stress]['operating_metrics']
            robust &= a['selected']['enabled'] and all(metric[c]['recall']>=base[c]['recall']-.01 and metric[c]['precision']>=base[c]['precision']-.01 for c in LABELS)
        pred=np.load(out/f'joint_temporal_seed{seed}/development_predictions.npz');frozen=np.load(out/f'frozen_temporal_seed{seed}/development_predictions.npz')
        for name,reference in [('baseline',pred['baseline_logits']),('frozen_temporal',frozen['logits'])]:intervals[f'seed{seed}_vs_{name}']=paired_video_bootstrap(m['splits']['development'],curves,pred['logits'],reference)
    uncertainty=all(v['percentile_95CI_pp'] is not None and v['percentile_95CI_pp'][0]>0 for v in intervals.values())
    result={'execution_complete':True,'execution_order':cfg['execution_order'],'results':results,'point_checks_passed':bool(point),'video_uncertainty_supported':bool(uncertainty),'synthetic_robustness_passed':bool(robust),'overall_checks_passed':bool(point and uncertainty and robust),'video_bootstrap':intervals,'no_localization_temporal_only_arm_run':False,'limitation':'no temporal-only control by user instruction: cannot separate benefits of temporal tuning alone from detection information causally','natural_absence_verified':False,'independent_test_claimed':False}
    atomic_json(out/'FINAL_SUMMARY.json',result)
    lines=['# Stage2 核心信息与时序头共同微调：按序实验结果','','只运行联合微调42→联合微调43→冻结时序对照42→冻结时序对照43。没有运行不带定位信息的时序头微调。720P、16帧、同一Stage1核心/上下文缓存。','','联合组：第一轮适配器预热，后五轮训练适配器+原事件时序Transformer+分类头。冻结组：六轮只训练适配器。DINO、Stage1定位头及原帧投影始终冻结；独立教师用于回退。','','|配置|种子|选中epoch|开发宏窗口AP|原模型AP|窗口P|事件R|','|---|---:|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f"|{r['arm']}|{r['seed']}|{r['selected']['epoch']}|{r['development_AP']['macro']*100:.3f}%|{r['baseline_development_AP']['macro']*100:.3f}%|{r['development_retuned']['macro_precision']*100:.3f}%|{r['development_retuned']['macro_recall']*100:.3f}%|")
    lines += ['',f'点估计检查：{bool(point)}；配对视频区间支持：{bool(uncertainty)}；启用模型的合成扰动检查：{bool(robust)}。', '', '检查点按历史校准集窗口宏AP选择；正常输入阈值仅校准集确定；原阈值P/R另存各RESULT。AP不是事件spotting mAP，误报单位为窗口/小时；开发视频不是独立盲测。','','不带定位信息的时序头微调未运行，不能把联合组相对原模型的全部收益归因于定位信息；相对冻结组的差异检验的是现有定位输入条件下的时序适配价值。','','空证据回退由独立冻结原模型保障。所选模型与末轮模型都在启用状态下评测半帧缺失、跨窗错配和局部时间反转；不能替代自然无球/无球门人工分层。','', '配对视频bootstrap宏窗口AP差（百分点）：']
    for name,v in intervals.items():lines.append(f"- {name}: {v['percentile_95CI_pp']}")
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines));atomic_json(out/'PIPELINE_COMPLETE.json',{'complete':True,'report':str(out/'FINAL_REPORT.md')})


def pipeline(out,cfg,m):
    lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cache=Path(cfg['core_cache_dir']);cache_sha=digest(cache/'manifest.json')
    provenance=json.loads((out/'run_provenance.json').read_text())
    adoption=json.loads((out/'cache_adoption.json').read_text())
    active={int(k.split('rank')[-1]):v for k,v in adoption['active'].items()};attempts={i:0 for i in range(4)}
    owned={}
    def integrity():
        for p,sha in provenance['sha256'].items():assert digest(p)==sha,'dependency changed: '+p
    def status(phase,**kw):
        value={'phase':phase,'supervisor_pid':os.getpid(),'updated_unix':time.time(),**kw};atomic_json(out/'pipeline_status.json',value)
        if phase in ['feature_cache','compacting']:atomic_json(cache/'pipeline_status.json',{**value,'training_queue_superseded_by':str(out)})
    def available_gpu():
        text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        free={int(line.split(',')[0]) for line in text.strip().splitlines() if int(line.split(',')[1])<256}
        return next((g for g in cfg['gpus'] if g in free),None)
    def alive(pid):
        try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
        except FileNotFoundError:return False
    def launch(script,args,tag,gpu=None):
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        log=(out/f'{tag}.log').open('a');p=subprocess.Popen([PYTHON,str(ROOT/script),*args],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        return p,log
    try:
        integrity()
        while not (cache/'COMPACT_COMPLETE.json').exists():
            complete=[]
            for rank in range(4):
                marker=cache/f'cache_rank{rank}_complete.json'
                if marker.exists():
                    assert json.loads(marker.read_text())['manifest_sha256']==cache_sha;complete.append(rank);continue
                item=active.get(rank)
                if item and alive(item['pid']):continue
                if rank in owned:
                    p,log=owned.pop(rank);p.wait();log.close()
                gpu=available_gpu()
                if gpu is None:continue
                if attempts[rank]>=2:raise RuntimeError(f'cache rank{rank} failed after two retries')
                p,log=launch('scripts/run_football_stage2_corepatch.py',['--output-dir',str(cache),'--phase','cache','--rank',str(rank)],f'cache_retry_rank{rank}',gpu)
                active[rank]={'pid':p.pid,'gpu':gpu};owned[rank]=(p,log);attempts[rank]+=1
                # Only start one retry per poll, avoiding a GPU allocation race.
                break
            if len(complete)==4:
                status('compacting');p,log=launch('scripts/run_football_stage2_corepatch.py',['--output-dir',str(cache),'--phase','compact'],'compact')
                while p.poll() is None:time.sleep(5)
                log.close();assert p.returncode==0,'compaction failed';continue
            status('feature_cache',active={f'cache_rank{k}':v for k,v in active.items() if k not in complete},completed_ranks=complete,training_order=cfg['execution_order']);time.sleep(5)
        for p,log in owned.values():p.wait();log.close()
        atomic_json(cache/'pipeline_status.json',{'phase':'cache_complete_training_superseded','new_experiment':str(out),'updated_unix':time.time()})
        integrity()
        # No same-priority parallelism: complete each configured run before next.
        for position,(arm,seed) in enumerate(cfg['execution_order']):
            tag=f'{arm}_seed{seed}'
            if (out/tag/'COMPLETE.json').exists():continue
            for attempt in range(1,3):
                gpu=available_gpu()
                while gpu is None:
                    status('waiting_gpu',next_run=tag,order_position=position+1);time.sleep(5);gpu=available_gpu()
                p,log=launch('scripts/run_football_stage2_joint.py',['--output-dir',str(out),'--phase','train','--arm',arm,'--seed',str(seed)],tag,gpu)
                while p.poll() is None:
                    status('training_and_evaluation',current_run=tag,order_position=position+1,total_runs=len(cfg['execution_order']),pid=p.pid,gpu=gpu,attempt=attempt);time.sleep(5)
                log.close()
                if p.returncode==0:break
                if attempt==2:raise RuntimeError(tag+' failed twice')
            integrity()
        status('final_report');report(out,cfg,m);status('complete',report=str(out/'FINAL_REPORT.md'))
    except Exception as e:
        status('failed',error=str(e));atomic_json(out/'PIPELINE_FAILED.json',{'not_complete':True,'error':str(e)});raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=DEFAULT);ap.add_argument('--phase',choices=['pipeline','train','report'],default='pipeline');ap.add_argument('--arm',default='joint_temporal');ap.add_argument('--seed',type=int,default=42);a=ap.parse_args()
    out=a.output_dir.resolve();cfg=json.loads((out/'config.json').read_text());m=json.loads((out/'manifest.json').read_text());assert cfg==m['config']
    if a.phase=='train':train(out,cfg,m,a.arm,a.seed)
    elif a.phase=='pipeline':pipeline(out,cfg,m)
    else:report(out,cfg,m)
if __name__=='__main__':main()
