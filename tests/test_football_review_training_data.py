import torch
import train_football_events as training
from football_review_data import clean_video, window_supervision, subtract_intervals


def row(identity, label, time, source='candidate', status='accepted', secondary=()):
    anchors = [{'source': 'gt', 'label': label, 'time_sec': time}] if source == 'gt' else []
    return {'id': identity, 'source_label': label, 'source_time_sec': time,
            'corrected_label': None, 'corrected_time_sec': time, 'status': status,
            'secondary_labels': list(secondary), 'payload': {'task_source': source,
            'support_start_sec': time - 5, 'support_end_sec': time + 5,
            'evidence_anchors': anchors, 'window_indices': [10]}}


def gt(identity, label, time):
    return {'event_id': identity, 'label': label, 'time_sec': time}


def test_confirmed_gt_ai_merge_preserves_reviewed_time_and_sources():
    left = row('gt_review', 'shot', 100, 'gt')
    left['corrected_time_sec'] = 100.4
    cleaned = clean_video([left, row('ai_review', 'shot', 101)], [gt('original', 'shot', 100)], 200)
    assert len(cleaned['events']) == 1
    event = cleaned['events'][0]
    assert event['time_sec'] == 100.4
    assert event['time_precise']
    assert set(event['source_ids']) == {'gt_review', 'ai_review'}
    assert event['lineage_gt_ids'] == ['original']
    assert event['sources'][1]['review_time_sec'] == 101


def test_distinct_dense_gt_and_cross_class_events_are_preserved():
    rows = [row('s1', 'shot', 100, 'gt'), row('s2', 'shot', 101, 'gt'), row('save', 'save', 100, 'gt')]
    cleaned = clean_video(rows, [gt('g1', 'shot', 100), gt('g2', 'shot', 101), gt('g3', 'save', 100)], 200)
    assert len(cleaned['events']) == 3
    assert not cleaned['conflicts']


def test_candidate_between_two_gt_is_quarantined_instead_of_nms():
    rows = [row('s1', 'shot', 100, 'gt'), row('s2', 'shot', 102, 'gt'), row('ai', 'shot', 101)]
    cleaned = clean_video(rows, [gt('g1', 'shot', 100), gt('g2', 'shot', 102)], 200)
    assert len(cleaned['events']) == 3
    assert len(cleaned['conflicts']) == 2
    assert not any(x['action'] == 'merge' for x in cleaned['audit'])


def test_uncertain_and_unchecked_are_separate_and_block_negatives():
    cleaned = clean_video([row('pending', 'shot', 100, status='needs_confirmation'),
                           row('unchecked', 'save', 100, status='unreviewed'),
                           row('rejected', 'shot', 100, status='deleted')], [], 200)
    assert {r['kind'] for r in cleaned['unknown']} == {'human_ambiguous', 'unchecked_candidate'}
    targets, masks, frame = window_supervision(cleaned, 95, 105)
    assert targets == [0, 0, 0]
    assert masks == [0, 0, 0]
    assert frame == [0, 0, 0]


def test_weak_positive_requires_support_coverage_and_disables_all_frame_losses():
    cleaned = clean_video([row('positive', 'shot', 100)], [], 200)
    targets, masks, frame = window_supervision(cleaned, 95, 105)
    assert targets[0] == masks[0] == 1
    assert frame[0] == 0
    assert window_supervision(cleaned, 98, 108)[1][0] == 0
    assert window_supervision(cleaned, 95, 105, temporal_evaluation=True)[1][0] == 0
    training.configure_label_schema({'task': {'label_schema': 'set_piece'}})
    logits = torch.randn(1, 4, 3, requires_grad=True)
    batch = {'targets': torch.tensor([targets]), 'label_masks': torch.tensor([masks]),
             'frame_targets': torch.zeros_like(logits), 'frame_target_masks': torch.zeros_like(logits),
             'frame_label_masks': torch.tensor([frame])}
    cfg = training.to_config({'train': {'frame_heatmap_loss_weight': 0.5,
                                        'frame_mil_loss_weight': 0.3, 'frame_rank_loss_weight': 0.2}})
    loss, components = training.frame_detection_loss({'frame_event_logits': logits}, batch, cfg, torch.device('cpu'))
    loss.backward()
    assert loss.item() == 0
    assert logits.grad.abs().sum().item() == 0
    assert components['frame_mil_loss'] == 0
    assert batch['label_masks'][0, 0] == 1


def test_trusted_background_loss_survives_and_unknown_regions_are_subtracted():
    cleaned = clean_video([row('background', 'shot', 100, status='deleted')], [], 200)
    assert window_supervision(cleaned, 95, 105)[1][0] == 1
    assert window_supervision(cleaned, 94, 104)[1][0] == 0
    assert subtract_intervals([[0, 40]], [[10, 20], [15, 25]]) == [[0.0, 10], [25, 40.0]]


def test_unreviewed_gt_remains_positive_but_never_frame_supervised():
    cleaned = clean_video([row('old_gt', 'shot', 100, 'gt', 'unreviewed')], [gt('original', 'shot', 100)], 200)
    assert not cleaned['events'][0]['confirmed']
    assert not cleaned['events'][0]['time_precise']
    assert window_supervision(cleaned, 95, 105) == ([1, 0, 0], [1, 0, 0], [0, 0, 0])


def test_set_piece_subtype_disagreement_is_not_merged():
    rows = [row('corner', 'set_piece', 100, 'gt', secondary=['corner']),
            row('kickoff', 'set_piece', 101, secondary=['kickoff'])]
    cleaned = clean_video(rows, [gt('original', 'set_piece', 100)], 200)
    assert len(cleaned['events']) == 2
    assert cleaned['conflicts']


def test_trusted_region_thresholds_ignore_unknown_high_score_before_nms():
    import numpy as np
    from football_online_evaluation import tune_online_event_thresholds
    training.configure_label_schema({'task': {'label_schema': 'set_piece'}})
    probs=np.array([[.7,.7,.7],[.99,.99,.99],[.2,.2,.2]])
    times=np.array([[5.,5.,5.],[7.,7.,7.],[15.,15.,15.]])
    masks=np.array([[1.,1.,1.],[0.,0.,0.],[1.,1.,1.]])
    metas=[{'source':'review','video_id':'v','sampled_clip_start':s,
            'sampled_clip_end':s+10,'online_gt_anchors':((5.,),(5.,),(5.,)),
            'review_trusted_regions':True} for s in [0.,2.,10.]]
    thresholds,diagnostics=tune_online_event_thresholds(probs,times,metas,training.LABELS,
        ['precision']*3,[.9]*3,masks,nms_radius_sec=5.,tolerance_sec=3.)
    assert np.allclose(thresholds,.7)
    for item in diagnostics.values():
        assert item['tp']==1 and item['fp']==0 and item['support']==1
        assert item['trusted_region_video_count']==1 and item['complete_video_count']==0
    result=training.online_event_metrics(probs,times,metas,np.array([.5]*3),masks,
        nms_radius_sec=5.,tolerance_sec=3.,capped_clip_sec=10.)
    for item in result['per_class'].values():
        assert item['tp']==1 and item['fp']==0 and item['fn']==0
        assert item['evaluation_scope']=='trusted_temporal_regions'


def test_gaussian_frame_targets_never_use_weak_timestamps():
    training.configure_label_schema({'task': {'label_schema': 'set_piece'}})
    event=training.FootballEvent(source='review',video_id='v',event_id='weak',event_type='',
        raw_label='shot',start_time=5.,end_time=5.,anchor_time=5.,labels=(1.,0.,0.),time_supervision=False)
    targets,_=training.gaussian_frame_targets([event],0.,10.,[1.,5.,9.],(1.,0.,0.),{}, {})
    assert targets.sum().item()==0


def test_final_case_generic_ai_and_specific_manual_duplicate_merge():
    from football_review_data import clean_final_events
    events=[{'source_id':'ai','semantic_label':'set_piece','time_sec':60.,'case_id':'case','lineage_gt_ids':[]},
            {'source_id':'human','semantic_label':'free_kick','time_sec':60.,'case_id':'case','lineage_gt_ids':[]}]
    cleaned,decisions=clean_final_events(events)
    assert len(cleaned)==len(decisions)==1
    assert cleaned[0]['semantic_label']=='free_kick'
    assert set(cleaned[0]['merged_source_ids'])=={'ai','human'}
    events[0]['lineage_gt_ids']=['gt1'];events[1]['lineage_gt_ids']=['gt2']
    assert len(clean_final_events(events)[0])==2


def test_remote_video_root_preserves_manifest_and_supervision():
    import hashlib,json,tempfile
    from pathlib import Path
    from football_review_data import load_records
    policy=clean_video([row('review_gt','shot',100,'gt')],[gt('g1','shot',100)],200)
    with tempfile.TemporaryDirectory() as directory:
        manifest=Path(directory)/'manifest.json'
        manifest.write_text(json.dumps({'videos':[{'video_id':'v1','split':'train','duration_sec':200,'video_path':'/original/v1.mp4','policy':policy}]}))
        before=manifest.read_bytes()
        cfg=training.to_config({'seed':42,'data':{'long_video':{'review_manifest':str(manifest),'review_manifest_sha256':hashlib.sha256(before).hexdigest(),'review_video_root':'/remote/raw_video_1080P'}},'video':{'clip_duration':10}})
        training.configure_label_schema({'task':{'label_schema':'set_piece'}})
        records,events=load_records(cfg,'train',training=training)
        assert records and all(r.video_path=='/remote/raw_video_1080P/v1.mp4' for r in records)
        assert records[0].review_manifest_path==str(manifest.resolve())
        assert any(e.time_supervision for e in events[('xbotgo_0608','v1')])
        assert manifest.read_bytes()==before
