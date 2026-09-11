# 足球事件检测全链路优化方案与首个实验(T1 帧级尾部排序)

日期:2026-09-10
状态:T1 已启动;T2/T3 为后续排队工作流
作者:qiuqi + kimi

## 0. 背景与诊断共识

当前锚点 E16(`vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_720p_fromlast_e8_20260829`):
DINOv3 ViT-L/16(冻结 + last-4 blocks LoRA r8)+ 4 层 cls_transformer(512d),720p 全图,16 帧/10s 窗。
test18 修复 GT 后(LOOV 自适应阈值,±3s):shot R 0.89/P 0.375,save R 0.83/P 0.241,set_piece R 0.87/P 0.25,
复核片长占比约 57%(产品门槛 recall≥90% 且复核<30%)。

已确认的核心矛盾(证据见 §5):**召回够用(proposal ceiling 0.96~0.97),高召回下精度差一个量级**,
机理是正负分数尾部重叠(tail_gap 全类为负:shot/save/set_piece ≈ -0.19/-0.26/-0.25,online 线更差 -0.33/-0.55/-0.67)。

三个外部条件(用户确认):
1. 训练数据人工返修进行中,后续有更高质量数据迭代;
2. 1~2 个月内数据量最多翻倍;
3. online-mode 采样对齐线下 dense 推理已试过,无收益(e20b/e21b 实测 tail_gap 未改善)。

### 关键代码事实(2026-09-10 核实)

- E16 锚点训练时 **`train.frame_det_loss_weight: 0`**:帧级稠密曲线头从未被直接监督,
  只靠 clip 级损失经池化间接塑形。这是 tail_gap 为负的最直接结构性解释。
- 现成 `frame_rank_loss`(train_football_events.py:14397)是"正帧 max vs 负帧 max"——
  优化最强正帧,恰好是尾部需求的反面。
- `online_global_tail_ranking_loss`(:13023)依赖 online simulation 元数据,离线数据不可用,
  且对应配置(`configs/football/dinov3_vitl16_online_global_tail_rank_highres720_from_best.yaml`)从未运行。
- 历史上 topk/OHEM/硬负例(other_action)均为负结果——但它们作用于 clip 级、且硬负标签语义同域,
  与"帧级曲线尾部塑形"不是同一干预,不能互相证伪。

## 1. 全链路方案总览

三条工作流,按"先对症已诊断瓶颈、再加新证据源、最后补物理上限"排序:

| 编号 | 工作流 | 对症瓶颈 | 新证据/新数据需求 | 预期收益 | 风险 |
|---|---|---|---|---|---|
| T | 帧级尾部排序目标(tail-first objective) | 尾部重叠(精度死因) | 无 | P@R85 +3~8pp | 低,单变量可控 |
| A | 音频/哨声融合(verifier 特征 + review 融合) | set_piece 判别证据不足 | 已有 64-bin log-mel 缓存 | free_kick R +5pp 级(已实测) | 低,工程整合 |
| D | 候选触发高帧率核验(8~10fps ±3s examiner) | 亚秒级动作不可见(save 物理上限) | 候选点高帧率抽帧 | save P 阶跃的唯一通道 | 中,需吸收 Stage2 负结果教训 |

组合关系:T 改的是稠密曲线本身的可分性,是 A 和 D 的地基(A/D 都消费 Stage-1 候选);
A 是低风险现成收益,可在 T 训练期间并行做工程整合;D 成本高、只打 save/shot 的亚秒证据,
等 T 的 go/no-go 结论出来后再启动。数据返修完成后,用同一协议重训 T 的最优配置,
新旧数据各跑一次,量化标注质量收益(预计 P 有机械性提升,参考 test18 修复后 shot P 0.202→0.271)。

### 与历史负结果的划界(为什么不重复失败)

- online-mode 无收益 ≠ 尾部排序无效:online 线从未直接监督帧级曲线尾部,变量混淆;
- Stage2 球 ROI 负结果是**空间**定位分支;D 工作流是**时间**加密,且只在候选点局部启用;
- 硬负例负结果是语义同域标签迫使去权重化共享证据;T 不引入任何新标签,只重排已有正负的相对顺序;
- E21b audio gate 无收益是 post-hoc 门控;A 工作流用 verifier 特征 + review 融合(已实测正向),不做 gate。

## 2. T1 实验规格(首个实验,最高优先级)

### 假设

在 E16 锚点之上,仅加一个帧级尾部排序损失,直接压低"负例高尾超过正例低尾"的重叠,
则 dense 验证上 tail_gap 收窄(向 0 靠近),且 recall floor 约束下 precision 提升。

### 单变量定义

基线 = E16 配置逐项不变(同 init_checkpoint=fromlast_e8/best.pt、同数据、同 lr、同选模规则);
唯一新增:`frame_tail_rank_loss`,小权重(0.1)+ 首 epoch 线性 ramp,clip 损失权重不变。
**不**开启完整 frame heatmap focal(避免"已训 checkpoint 上重开帧监督=倒退课程"的历史失败模式)。

### 损失设计

对每个样本(10s 窗 × 16 帧)× 每类:
- 正帧集:`frame_target_masks & frame_targets > 0.5`(事件 anchor ±~0.94s 内);
- 负帧集:`frame_target_masks & frame_targets <= 0`(ignore radius 之外的背景帧);
- 正例低尾:soft-min,`pos_tail = -τ·logsumexp(-s_pos/τ)`;
- 负例高尾:soft-max,`neg_tail = τ·logsumexp(s_neg/τ)`;
- 窗内项:`L_in = softplus((margin - (pos_tail - neg_tail))/β)·β`;
- 跨窗项(batch 内,含负窗):同类所有正帧的 soft-min vs 所有背景帧的 soft-max,`L_out` 同式;
- 总:`L = w_ramp · (L_in + L_out) / 2`,逐类计算后均值;label_mask=0(untrusted/unknown)的类跳过;
- 默认:margin=1.0,τ=0.5,β=1.0,w=0.1,ramp 1 epoch。

诊断量每 epoch 落盘:`frame_tail_pos_neg_gap`(batch 内均值)、有效项计数;
验证侧沿用现成 `confidence_separation.tail_gap`(train_football_events.py:12058)做先行指标。

### 实现偏差记录(2026-09-10,启动前决策)

冒烟发现:字面公式下 L_in(窗内项)结构性不激活——`gaussian_frame_targets()` 对含该类 anchor 的窗
把整列写成严格 >0 的热图,窗内不存在 `targets<=0` 的负帧。决策:**设 `frame_tail_rank_neg_threshold=0.05`**
(正式配置已改),即窗内热图 ≤0.05 的上下文帧(shot 约距 anchor ≥2s)参与窗内负例高尾排序。
理由:线上 FP 主形态是事件近邻的持续高分平台,排序约束是相对的(只要求低于正帧低尾),
不同于 heatmap focal 把其当绝对负例;0.05 与代码库现有 `frame_rank_neg_threshold` 先例一致。
冒烟证据(40 micro-step,单卡子集):loss 非零有限有梯度,`frame_tail_pos_neg_gap`≈-0.25~-0.45,
ramp 行为正确,诊断全部落盘,clip loss 与评测链路无回归。

### 决策门(预注册,防止事后解释)

在 val15 dense 口径、recall floor shot 0.85/save 0.80/set_piece 0.80 下,与 fromlast_e8 锚点同口径对比:
- **先行指标**(epoch 2 即可判):三类 tail_gap 至少两类改善 ≥0.05;若不满足 → 假设证伪,
  T 工作流停止,资源转 D(时序密度);
- **主指标**(epoch 4):tuned macro precision 提升 ≥+2pp 且各类 recall 不低于 floor;
- **副作用红线**:sampled-clip mAP 下降 >1pp 或 recall shortfall 出现 → 判失败回滚。

### 工程参数

- GPU:2 张空闲卡(5090 ×2),DDP;per_gpu_batch 2、grad_accum 20,保持 effective batch=80 与锚点一致;
- epochs 4(短周期快速拿 go/no-go;约 8h/epoch,总计 ~32h);
- 输出:`outputs/football_events/vitl16_d7c_fullimage_720p_fromlast_e8_frame_tail_rank_v1_20260910/`;
- 配置:`configs/football/vitl16_720p_fromlast_e8_frame_tail_rank_v1.yaml`;
- 启动脚本:`scripts/run_frame_tail_rank_v1_20260910.sh`。

## 3. T2/A1/D1 排队预案(触发条件明确,不在本次执行)

- **T2**:T1 若 go,权重/温度网格(w ∈ {0.05,0.1,0.3},τ ∈ {0.25,0.5})小扫一档;
  同时把 tail 诊断加进 checkpoint 选择分(当前选择分只看 tuned macro P);
- **A1**(可与 T1 训练并行做工程):哨声概率作为候选 verifier 特征 + free_kick/corner 的
  review-only 区间融合(复用 9/9 评估中已实测 R +5.8pp 的路径),先落在 dense 评测脚本层,
  不进模型;模型侧 audio token 等 T 结论后再议;
- **D1**(T 若 no-go 则提前):候选峰 ±3s 内 8~10fps 抽帧,冻结 DINO + 轻量时序头,
  只在候选点推理(覆盖 <5% 时间轴,算力可控);先在 6 视频 OOF 上证明 save 类
  tail gap 收窄,再上全量。明确不做空间 ROI token 注入(Stage2 已证伪)。

## 3.5 并行开发记录(2026-09-10,与 T1 训练同时)

价值评估后,A1 升级为 **Verifier V2** 并立即开发(CPU-only,不占 T1 训练卡),理由:
V1 候选 verifier 是项目历史上最大单一正向信号(6 视频 OOF micro P 0.2166→0.4378),但受困于
数据规模(6~15 视频)与特征宽度(7 维/26 维)。full166 稠密缓存
(`outputs/football_full_review/full166_fromlast_e8_best_dense_s5_20260902/`,E16 同 checkpoint)
覆盖 129 个训练视频 + val15 + test18,使 verifier 训练数据放大 ~20 倍。

V2 规格:
- 候选:window_predictions.csv 融合分曲线逐类低阈值(0.05~0.10)提峰 + 类 NMS(shot/save 3s,set_piece 5s);
- 特征组:分数系(clip/response/global prob、logit、跨类 margin)+ 峰形系(prominence、半峰宽、
  升降斜率、平台时长,由分数时间序列计算)+ 时序上下文系(候选密度、窗口冗余覆盖数;
  save 条件化:过去 6s/12s 最大 shot 概率、距上个 shot 候选时间)+ 视频级 median logit
  + 音频系(哨声 peakiness 在 [-5s,+10s] 的 max/mean、whistle activity 标志;训练侧音频特征
  用 scripts/build_audio_feature_index.py 现补);
- 训练:train129 GroupKFold(5-fold,by video)OOF + StandardScaler + LogisticRegression
  (class_weight=balanced, C=0.1,沿用 V1 已验证配方);特征组消融(OOF 口径);
- 定阈与迁移:val15 上按 recall floor(shot 0.85/save 0.80/set_piece 0.80)选 per-class 阈值,
  冻结后一次性迁移 test18(GT 用 test18_final_repaired_v2,与 0909 评估同口径可比);
- 对照基线:同候选集上分数阈值 baseline 在相同召回下的 P;当前运行点(frozen transfer
  shot P 0.412/R 0.802,save 0.285/0.751,set_piece 0.252/0.865);
- 产物:outputs/football_candidate_verifiers/v2_full166_20260910/。

预期:V1 在 6 视频上已证明 P 翻倍信号;数据放大 20 倍 + 峰形/时序/音频特征,
目标 val15/test18 上 P@Rfloor 相对当前运行点提升 ≥+5pp 且不掉 recall。

**V2 结果(2026-09-10 完成,outputs/football_candidate_verifiers/v2_full166_20260910/REPORT.md):未达标,判负。**
test18 冻结迁移(window_overlap tol3,与 0909 同口径):自适应迁移下 shot P 0.423/R 0.836、
save 0.246/0.897、set_piece 0.347/0.718,对照 0909 运行点(0.412/0.802、0.285/0.751、0.252/0.865)
与同候选集 peak_score baseline,最大增益仅 set_piece +4.3pp P 且有 recall 让步;shot +1.1pp。
**机理结论(重要):window_overlap 口径下所有特征都派生自同一条分数曲线,LR 重排空间有限;
V1 的翻倍增益依赖 frame logits 特征,而 full166 缓存未落 frame logits。**
V2.1 方向:重跑稠密缓存带 --save-frame-event-logits(GPU 推理,等卡空闲),或直接用 T1 新曲线。
过程资产:train129/val15 的 5Hz 音频特征与 whistle activity 已全部补齐
(outputs/football_audio_features/{train129,val15}_5hz_20260910/),音频特征组在 OOF 上对
shot 有 +0.8pp、set_piece +0.3pp 的小幅正贡献,不足以单独成立。
Verifier 路线暂停,优先级让位于 T1(T 工作流直接塑造帧级曲线,正是 V2 暴露的缺环)。

## 4. T1 实验结果与证伪结论(2026-09-11 04:30)

**T1 未通过预注册决策门,已于 epoch 2 后停止(保留 2 个 epoch checkpoint)。**

| tail_gap | T1 ep1 | T1 ep2(满权重) | 锚点 | 判定 |
|---|---|---|---|---|
| shot | -0.268 | -0.229 | -0.232 | 持平 |
| save | -0.571 | -0.519 | -0.421 | 变差 -0.097 |
| set_piece | -0.348 | -0.185 | -0.188 | 持平 |

tuned macroP 0.442 vs 锚点 0.454(-1.2pp),mAP 0.570 vs 0.576。零类 tail_gap 改善 ≥0.05 → 先行指标证伪。

**机理结论(比结果本身更重要)**:训练 batch 内 gap 持续改善(-0.429→-0.376),即损失确实在优化
其目标,但改善不迁移到验证集尾部。结合 online-mode 无收益的先例,判定:**训练窗口分布
(每窗一事件 + curated 负窗)里不存在长视频连续背景中真正伤害精度的负例高尾,任何只在
该分布上做的损失/采样改动都无法触及问题本身**。后续方向必须满足其一:(a) 在连续流上
监督/约束(dense-stream supervision);(b) 引入新的判别证据(高帧率时序、音频),在候选级做决策。

**资源转向**:T 工作流停止(含 T2 超参计划);GPU 转给 full166 frame-logits 稠密缓存
(scripts/run_full166_fromlast_dense_framelogits_20260910.sh,V2.1 与 D1 共用资产);
D 工作流(候选点高帧率核验)提前启动开发,见 §5。

## 5. D1 候选点高帧率核验器(2026-09-11 启动)

- **假设**:save/shot 的判别证据在亚秒级(触球、扑救出手),1.6fps 特征不可见;在候选峰 ±3s
  以 8fps 提取冻结 DINO 特征,轻量时序头做候选级二分类,补足 V2 窗口特征缺的帧级证据。
- **数据**:复用 V2 candidates.csv(outputs/football_candidate_verifiers/v2_full166_20260910/),
  train129 训练 / val15 定阈 / test18 冻结迁移,与 V2 完全同协议,增量对比(V2 特征 ± D1 特征)。
- **特征**:候选 ±3s × 8fps = 48 帧,冻结 DINO ViT-L/16(720p),CLS+patches 均值 → [48,1024] fp16,
  不存原始帧(存储约束);GPU 提取脚本就绪后排队等卡。
- **头**:2 层 Transformer(256d)候选级二分类,per class;target 与 V2 相同(±3s/8s 忽略带)。
- **决策门**:OOF 上 save P@R80 提升 ≥+3pp 才继续投入;否则 D 降级为 save 专线后处理思路。

## 5.5 D2 主模型时序密度化(2026-09-11 设计,触发条件 = D1 过门)

**动机**(用户提出,与既有证据一致):当前 16 帧/10s 等间隔采样是"等信息密度",但事件在时间上
高度非均匀——动作爆发在亚秒级,上下文在十秒级。E1–E5 头消融全部无收益 + save 最差 +
T1 证伪,三条证据共同指向:信息在采样环节已丢失,后端无法补救。

**形态选择**:
- A 局部早融合(tubelet):8 帧(1s)轻量时空融合 → 1 个 tubelet token,10s=10 token 进时序头。
  运动信息在进 transformer 前完成编码;token 更短,长上下文更便宜。football_longform_v2 的
  LF-A2 骨架(0.5s tubelet + dilated blocks + 126s 感受野)可直接复用。
- B 多速率双流(SlowFast 式):慢流语义 + 快流运动,横向连接。
- C 候选触发局部加密:即 D1,作为主模型的廉价探针先行。
首选 A(tubelet),理由是 token 经济 + LF-A2 现成骨架。

**工程前提**:backbone 冻结+LoRA → 可预计算全量 8fps 稠密特征库(train+val ≈ 560 万帧
≈ 45 GPU 小时),训练只读缓存,架构迭代成本极低。

**执行序列**:D1 出结论(go/no-go)→ go 则建特征库并训 M1(tubelet 融合头,对照 E16 同协议);
no-go 则时序密度路线整体关闭,转连续流监督/工程侧。

## 6. 数据迭代配合

- 返修数据到位后:用 T 的最优配置在旧数据/新数据各训一次(同 seed、同 epoch 数),
  dense 口径对比,量化标注质量收益;若新数据收益 >2pp,后续实验全部切新数据;
- 数据扩增(最多翻倍)到位后:优先补 save 与 set_piece 正例(当前 P 最低的两类),
  采样比例向这两类倾斜;翻倍后重跑 T 最优配置一次即可,不重新扫超参;
- 评测卫生:本轮起冻结一个新 holdout(8~10 场,不参与任何选模/定阈),val15/test18 降级为开发集。

## 5. 证据索引

- 瓶颈审计:`docs/dino_stage1_precision_bottleneck_audit_20260824.md`(tail gap、LoRA bug、窗口重复 FP)
- 当前最好口径:`outputs/football_eval_runs/vitl16_d7c_fullimage_latest_split_720p_best_epoch3_test18_finalgt_v2_20260909/`
- online 线 tail_gap 未改善:outputs/football_events/vitl16_online_simulation_e20b/e21b 的 best_metrics.json
- E16 锚点配置:`outputs/football_events/..._720p_fromlast_e8_20260829/config.yaml`(frame_det_loss_weight=0)
- 哨声实测收益:`outputs/football_eval_runs/.../setpiece_whistle_subtype_recall_loov_tol3.json`(free_kick R 0.843→0.901)
- Stage2 空间分支负结果:`outputs/football_localization_stage2/720p_optional_ball_roi_from_stage1e2_20260909/FINAL_REPORT.md`
- 硬负例负结果:`docs/README_FOOTBALL_EVENT_EXPERIMENTS.md` §26
