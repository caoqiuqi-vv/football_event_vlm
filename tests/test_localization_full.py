import numpy as np
import torch
from football_localization_full import CoverageSampler,global_localization_loss
from scripts.train_football_localization_stage1 import spatial_targets


def test_full_coverage_and_exact_resume():
    lengths=[11,5,113,24];batch=8;world=5
    samplers=[CoverageSampler(lengths,batch,r,world,42,1,block=13) for r in range(world)]
    assert len({len(s) for s in samplers})==1
    values=np.concatenate([s.values[s.values>=0] for s in samplers])
    assert np.array_equal(np.sort(values),np.arange(sum(lengths)))
    for rank,s in enumerate(samplers):
        resumed=CoverageSampler(lengths,batch,rank,world,42,1,block=13,start_step=1)
        assert np.array_equal(resumed.values,s.values[batch:])
    next_order=CoverageSampler(lengths,batch,0,world,42,2,block=13)
    assert not np.array_equal(next_order.values,samplers[0].values)


def test_masked_padding_and_equal_class_loss():
    boxes=torch.zeros(3,2,16,4);boxes[:2,0,0]=torch.tensor([.4,.4,.41,.41]);boxes[1,1,0]=torch.tensor([.2,.2,.4,.5])
    weights=torch.tensor([[.8,0],[.6,.9],[0,0]])
    targets=spatial_targets(boxes,weights>0)
    logits=torch.randn(3,3600,2,requires_grad=True)
    loss=global_localization_loss(logits,targets,weights)
    ce=-(targets*logits.log_softmax(1)).sum(1)
    expected=((ce[:2,0]*weights[:2,0]).sum()/1.4+ce[1,1])/2
    assert torch.allclose(loss,expected,atol=1e-6)
    loss.backward();assert torch.count_nonzero(logits.grad[2])==0
    assert torch.count_nonzero(logits.grad[0,:,1])==0


if __name__=='__main__':
    test_full_coverage_and_exact_resume();test_masked_padding_and_equal_class_loss();print('full localization assertions PASS')


def test_report_ratios_do_not_confuse_frame_and_video_means():
    from scripts.finalize_football_localization_full import compare
    dtype=np.dtype([('video','i4'),('valid','?',(2,)),('error','f4',(2,2)),('inside','?',(2,2))])
    rows=np.zeros(110,dtype=dtype);rows['video'][10:]=1;rows['valid'][:,0]=True;rows['error'][:]=100
    rows['error'][0,0,0]=0
    rows['error'][10:20,1,0]=0
    result=compare(rows,['short','long'])['ball']
    assert abs(result['pooled_delta_pp']-(-9/110*100))<1e-8
    assert abs(result['equal_video_mean_delta_pp'])<1e-8


def test_incomplete_run_cannot_generate_final_report():
    import tempfile,json
    from pathlib import Path
    from scripts.finalize_football_localization_full import finalize
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);(p/'manifest.json').write_text(json.dumps({'config':{'epochs':2}}))
        try:finalize(p)
        except FileNotFoundError:pass
        else:raise AssertionError('incomplete run was accepted')
        assert not (p/'FINAL_REPORT.md').exists()


def test_final_report_pipeline_with_complete_numeric_fixture():
    import tempfile,json
    from pathlib import Path
    from football_localization_full import digest
    from scripts.finalize_football_localization_full import finalize
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);(p/'source.pt').write_bytes(b'synthetic-fixture-source');(p/'warm.pt').write_bytes(b'synthetic-fixture-head')
        np.save(p/'frames.npy',np.arange(3));desc=[{'video_id':v,'array_dir':str(p),'array_sha256':{'frames':digest(p/'frames.npy')}} for v in ('v0','v1')]
        cfg={'epochs':1,'checkpoint':str(p/'source.pt'),'warmup_checkpoint':str(p/'warm.pt'),'min_patch_cosine':.98,'min_cls_cosine':.98}
        manifest={'config':cfg,'counts':{'train':{'unique_annotated_rgb_frames':3}},'splits':{'train':desc,'val':desc}}
        (p/'manifest.json').write_text(json.dumps(manifest));sha=digest(p/'manifest.json')
        (p/'TRAINING_COMPLETE.json').write_text(json.dumps({'epochs':1,'coverage_pass':True}));np.save(p/'coverage_epoch_001.npy',np.ones(3,dtype=np.int32))
        (p/'coverage_epoch_001.json').write_text(json.dumps({'missing':0,'duplicates':0,'manifest_sha256':sha}))
        (p/'source_provenance.json').write_text(json.dumps({'snapshot':str(p/'source.pt'),'checkpoint_sha256':digest(p/'source.pt'),'warmup_snapshot_sha256':digest(p/'warm.pt')}))
        torch.save({'manifest_sha256':sha,'epoch':1},p/'best_adapt.pt');torch.save({'manifest_sha256':sha,'epoch':0},p/'best_control.pt')
        dtype=np.dtype([('index','i8'),('video','i4'),('valid','?',(2,)),('peaks','i4',(2,2)),('error','f4',(2,2)),('inside','?',(2,2)),('stats','f4',(4,))])
        r=np.zeros(3,dtype=dtype);r['index']=np.arange(3);r['video']=[0,0,1];r['valid']=True;r['error']=100;r['error'][:2,0,0]=1;r['error'][0,1,0]=1;r['inside'][:2,0,1]=True;r['inside'][0,1,1]=True;r['stats']=[.001,.001,.999,.999]
        np.save(p/'val_epoch_000_predictions.npy',r);np.save(p/'val_epoch_001_predictions.npy',r);r['valid'][:,1]=False;np.save(p/'expert_selected_predictions.npy',r)
        (p/'expert_reference_manifest.json').write_text(json.dumps({'frames':3,'descriptors':desc}))
        s=finalize(p);assert s['execution_complete'] and not s['real_detection_accuracy_verified'] and not s['stage2_started']
        assert (p/'FINAL_REPORT.md').is_file() and (p/'PIPELINE_COMPLETE.json').is_file()
