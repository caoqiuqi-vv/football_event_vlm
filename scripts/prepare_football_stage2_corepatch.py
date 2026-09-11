#!/usr/bin/env python
"""Freeze the predeclared factorial protocol, reusing existing splits verbatim."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from football_localization_full import digest,atomic_json
from scripts.run_football_stage2_corepatch import DEFAULT
BASE=ROOT/'outputs/football_localization_stage2/720p_optional_ball_roi_from_stage1e2_20260909'

def main():
    out=DEFAULT;out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():print('already prepared',out);return
    assert shutil.disk_usage(out).free>250_000_000_000,'need >=250GB for cache plus compact arrays and working margin'
    old=json.loads((BASE/'config.json').read_text());m=json.loads((BASE/'manifest.json').read_text())
    cfg={**old,'output_dir':str(out),'base_dir':str(BASE),'eval_batch_size':64,'hidden':128,
         'selection':'calibration_macro_window_AP','seeds':[42,43],'epochs':6,'batch_size':64,
         'arms':{'roi_late':{'representation':'roi','source':'stage1','fusion':'late'},
                 'core_late':{'representation':'core','source':'stage1','fusion':'late'},
                 'roi_temporal':{'representation':'roi','source':'stage1','fusion':'temporal'},
                 'core_temporal':{'representation':'core','source':'stage1','fusion':'temporal'},
                 'frozen_core_temporal':{'representation':'core','source':'frozen','fusion':'temporal'},
                 'ordinary_core_temporal':{'representation':'core','source':'ordinary','fusion':'temporal'}},
         'threshold_policy':'calibration-tuned normal input; original thresholds reported separately and used on explicit all-null fallback',
         'scope':'720P 16-frame frozen Stage1 and original event source; exact core tokens plus spatial context; zero-initialized adapter; same historical calibration/dev split; no new test use'}
    m['config']=cfg;m['base_manifest_sha256']=digest(BASE/'manifest.json')
    atomic_json(out/'config.json',cfg);atomic_json(out/'manifest.json',m)
    protocol='''# Stage2 核心 patch 与时序融合配对实验（预注册）

用户授权：保持720P、16帧，核心3×3原始patch、独立上下文网格、时序交叉注意力、多候选与空状态。首轮只用足球；不增加帧数、不用球门存在作筛选。

沿用上一轮全部32,072训练窗口、7视频校准5,618窗、8视频开发5,919窗，43,585唯一窗口。视频、帧号、标签、未知掩码、划分逐项复用。历史使用的开发集不是独立盲测；18视频测试集不参与。

每帧取两个分離候选峰，各直接gather 3×3原始patch（包含中心），另保留11×11邻域ROIAlign的2×2上下文token，合计26个token。上下文采样ratio=0按区域自适应；核心不经ROIAlign/平均。边缘core位置越界显式mask，不复制边界patch充数。描述含xy、相对位置、尺度、核心/上下文类型、相对热图响应、熵/峰差与候选编号；时间独立编码。

适配器以原模型512维帧token为query，对前/当前/后一帧的候选进行软注意力，含NULL；真实时间随key保留，不强制轨迹，也不做坐标插值。无球/无球门是正常状态，不能从空间softmax峰值推断存在。原全局通路始终保留。若所有候选显式无效，输出原分数且使用原阈值。自然无球鲁棒性仍需人工分层验证。

六组：roi_late、core_late、roi_temporal、core_temporal、frozen_core_temporal、ordinary_core_temporal。前四组构成信息表示×融合位置的2×2因素设计。后三种temporal对比定位信息来源。每组同一适配器、同一可训练参数量，种子42/43，各6轮。late 是分类头前的证据读出，转换成加到原分数的残差；temporal是在原4层事件Transformer之前加入交叉注意力结果。旧实验的随机独立时序网络仅为历史参考，此次roi组重新训练以匹配容量。

原事件模型和Stage1全部冻结，只有适配器训练。输出投影严格零初始化。两种融合均用相同源模型的配对计算得到变化量，确保初始化和显式全空回退逐项相等。原源模型为用户指定720P fromlast best的只读快照，定位为Stage1 epoch2，对照定位头为epoch1。

缓存重新解码同一批原始720P帧；较高分辨率来源沿用原事件INTER_CUBIC预处理。每个窗口重新计算的原全局特征、旧ROI向量及热图描述必须与旧缓存逐项一致，否则停止，不混合不一致输入。三种core特征使用FP16缓存，边界mask和核心原始索引同时保存；这是缓存量化，不声称原生FP32。

优化沿用AdamW lr2e-4、batch64、weight_decay0.01，余弦衰减、原类别权重[1,2.5,4]、残差L2 0.01、跨窗口错误候选残差约束0.05。整窗/帧/候选缺失概率0.15/0.20/0.10；按两个候选组drop，以保持ROI/core间扰动可比。缺失不改变事件标签。新方案不追加额外存在性、轨迹伪标签或过强语义KL。

检查点按校准集宏窗口AP选择，epoch0原模型作为候选。AP以10秒窗、5秒步长、±2秒容差内是否有事件定义窗口正例；不是事件spotting mAP。正常输入阈值只在校准集按召回下限[.84,.80,.72]确定，开发集禁止选点；同时报告沿用原阈值的P/R。FP/小时指错误窗口，不是去重事件实例。

通过正常输入检查要求两个种子的core_temporal都：开发宏AP高于原模型及roi_temporal/core_late/冻结定位/普通区域；校准阈值下开发宏P高于原模型、每类R不低于原模型1pp且错误窗口/小时不增加。按视频汇报500次配对重采样区间；另检查core_temporal相对各对照的宏AP差区间下界是否均大于0。所选启用模型在半数帧缺失/跨窗错误候选下，各类P/R不得低于原模型超过1pp。分别报告点估计、区间支持和合成鲁棒性，全部通过也不等于自然无球或独立测试泛化已通过。启用分支的缺失、跨窗口置换、时间反转单独报告，不能以关闭分支代替鲁棒性证据。

运行将自动完成四卡缓存、覆盖/一致性核验、12组训练、开发评测、最终报告。全部完成前不输出正向结论。源码/配置/输入模型与数据清单固定摘要；不修改原实验或原模型。
'''
    (out/'PROTOCOL.md').write_text(protocol)
    print(out)
if __name__=='__main__':main()
