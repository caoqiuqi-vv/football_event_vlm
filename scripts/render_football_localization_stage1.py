#!/usr/bin/env python
"""Scientific diagnostic: sampled 720P frames with teacher boxes and model peaks."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_football_localization_stage1 import LocalizationModel, Frames


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output-dir",required=True);a=ap.parse_args();out=Path(a.output_dir)
    state=torch.load(out/"best.pt",map_location="cpu",weights_only=True)
    cfg=state["config"];torch.set_num_threads(1)
    model=LocalizationModel(cfg).cuda().eval()
    params=dict(model.backbone.named_parameters())
    assert set(state["backbone_lora"])=={n for n,p in params.items() if p.requires_grad}
    with torch.no_grad():
        for name,value in state["backbone_lora"].items():params[name].copy_(value)
    model.adapt_head.load_state_dict(state["adapt_head"]);model.control_head.load_state_dict(state["control_head"])
    records=json.loads((out/"manifest.json").read_text())["records"]["val"]
    predictions=json.loads((out/f"predictions_epoch_{state['epoch']:03d}.json").read_text())
    chosen=[]
    for c in (0,1):
        metric="within_16px" if c==0 else "peak_inside_box"
        cases=[r for r in predictions if r["class"]==c]
        for category,predicate in [("improved",lambda r:r['adapt'][metric] and not r['control'][metric]),("regressed",lambda r:r['control'][metric] and not r['adapt'][metric]),("both_correct",lambda r:r['control'][metric] and r['adapt'][metric]),("both_missed",lambda r:not r['control'][metric] and not r['adapt'][metric])]:
            matching=[r for r in cases if predicate(r)]
            if matching:chosen.append((matching[0]['index'],category))
    dataset=Frames(records);fig,axes=plt.subplots(4,2,figsize=(16,18));details=[]
    for ax in axes.flat:ax.axis('off')
    for ax,(idx,category) in zip(axes.flat,chosen):
        b=dataset[idx];r=records[idx];frame=b['frame'];c=r['object']
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):result=model(frame[None].cuda(),adapt=bool(state['metrics']['adapt']))
        ax.imshow(frame.permute(1,2,0).numpy())
        for box in r['boxes']:
            x1,y1,x2,y2=np.array(box)*[1280,720,1280,720];ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='lime',linewidth=2))
        for arm,color,marker in [('control','cyan','x'),('adapt','red','+')]:
            peak=int(result[f'{arm}_logits'][0,:,c].argmax());x=(peak%80+.5)*16;y=(peak//80+.5)*16
            ax.plot(x,y,marker=marker,color=color,markersize=13,markeredgewidth=2)
        ax.set_title(f"{'ball' if c==0 else 'goal'} / {category} / {r['video_id']} @ {r['time']:.2f}s",fontsize=9)
        details.append({'index':idx,'category':category,'video_id':r['video_id'],'time':r['time']})
    fig.suptitle(f"Stage1 epoch {state['epoch']} | green: automatic teacher | cyan: frozen probe | red: adapted probe\nIllustrative selected cases, not representative accuracy estimates",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'localization_examples.png',dpi=130);plt.close(fig)
    (out/'localization_examples.json').write_text(json.dumps(details,indent=2));print(out/'localization_examples.png')


if __name__=='__main__':main()
