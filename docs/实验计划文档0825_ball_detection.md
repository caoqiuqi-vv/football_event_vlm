# 足球关键事件：Ball Detection 辅助监督与特征路由实验计划

更新时间：2026-08-25

状态：方案冻结前评审稿。本文档定义实验假设、数据生产、网络结构、监督语义、对照实验、评测协议和停止条件；在 Phase 0 完成前不启动全量伪标签生产。

## 1. 目标与边界

### 1.1 最终目标

让 DINOv3 的逐帧表示显式保留足球小目标证据，并让该证据真正进入 shot/save 的时序判别，从而在完整长视频 dense 扫描中：

- 提高 `Precision @ Recall >= 85%`；
- 尽可能提高 `Precision @ Recall >= 90%`；
- 降低 `FP / 90min`；
- 降低跨类别、跨时间去重后的人工参与时长；
- 不因 detector 漏检而损伤事件召回；
- 改善跨视频泛化，而不是只提高训练集 detection loss。

### 1.2 不做什么

- 不把当前漏检严重的 detector index 当作完整 GT；
- 不把“未检测到球”直接监督成“画面没有球”；
- 不要求 patch-size 16 的 DINO 精确回归亚 patch 级足球框；
- 不把 ball presence 当作 shot/save 的充分条件；
- 不在首轮同时改变 LoRA depth/rank、时序头、Online-simulation、输入分辨率和 detection routing；
- 不使用 external18 选择 checkpoint、阈值、loss 权重或伪标签阈值。

## 2. 核心假设

### H1：表征假设

当前 DINO 表示擅长人和场地等大目标，但足球经常小于一个 ViT patch，ball evidence 在共享 CLS/attentive pooling 中被背景稀释。可信的 ball spatial supervision 可以提高对应 patch/query 对足球的可分性。

### H2：路由假设

仅增加 detection auxiliary loss 不保证事件指标提升。只有 ball evidence token 进入 class-conditioned per-frame state，并参与 Temporal Encoder，事件 loss 才能利用该证据。

### H3：领域假设

当前运动相机视频的视角、压缩、模糊和球尺度与外部检测数据不同。强但慢的足球检测模型在当前训练视频上生成的高质量 pseudo positives，比只使用外部检测数据更接近下游域。

### H4：安全假设

teacher 漏检是不对称风险。伪标签应采用 positive/unknown/trusted-negative 三态语义；如果把漏检帧当负样本，ball-aware训练会反向损害 shot/save。

## 3. 总体技术路线

```text
外部真实ball detection GT ───────────────┐
                                          │
当前域HQ视频 -> 强慢Detection Teacher -> pseudo boxes/conf/track
                                          │
                                          v
                              可信监督与unknown mask
                                          │
Frame -> DINOv3 patch tokens + CLS         │
                 │                         │
                 ├-> Ball heatmap/query head <─ detection loss
                 ├-> Goal heatmap/query head（control/可选）
                 └-> Global frame feature
                            │
                   class-conditioned routing
                            │
            shot/save/set_piece state token S_t^c
                            │
                   Temporal Stem/Encoder
                         /        \
                 Dense curve     Clip head
                            │
                      event losses
```

训练完成后，慢检测器不进入线上主链路。线上由 DINO 自己产生 ball heatmap/token；若后续证明高分辨率局部重编码必要，再单独立项 sparse high-res evidence，不与首版混合。

## 4. 数据来源与隔离

### 4.1 数据源

1. 当前足球事件训练视频：train split only。
2. 外部真实足球检测数据：路径、license、类别定义和视频级split在启动前登记。
3. 强慢足球检测 Teacher：checkpoint、代码版本、输入尺度、tile策略、推理速度在 Phase 0 登记。
4. 当前已有 detection index：只作为候选和覆盖率参照，不直接视为完整GT。

### 4.2 数据隔离

- pseudo label生产只允许处理 train 视频；
- val15只允许做人审Teacher质量、ball probe和下游模型验证，不能用于训练；
- external18保持 untouched，仅在方案冻结后报告一次；
- 所有检测数据按原始视频或比赛划分，禁止相邻帧跨 train/val 泄漏；
- 如果外部检测数据与现有事件视频同源，必须按 video ID 去重。

### 4.3 原图坐标语义

所有框和中心点必须保存为原始视频坐标系下的归一化坐标：

```text
x_norm = x_pixel / original_width
y_norm = y_pixel / original_height
```

索引必须保存：

- `video_id`、原始 `frame_id`、`timestamp_sec`；
- 原始 `width/height/fps`；
- normalized `xyxy`；
- teacher confidence；
- detection source/full-frame/tile/secondary；
- track ID与temporal consistency；
- pseudo label version和teacher checkpoint hash。

## 5. Phase 0：Teacher 与数据审计

### 5.1 必须回答的问题

- 强Teacher在当前运动相机域是否真的明显优于旧index？
- 小球、模糊、遮挡、球门附近、人群密集处的recall分别是多少？
- Teacher单帧速度、显存和多卡吞吐是多少？
- 外部真实检测数据与当前域的ball像素尺寸分布是否一致？

### 5.2 人审集

从 train/val15 分层抽取1000至2000帧：

- 30% shot/save附近 ±3秒；
- 20% set_piece准备及触球附近；
- 20% 普通控球/传球/回传；
- 15% 球门附近、人群密集、遮挡；
- 15% 随机背景。

标注：ball是否可见、中心/框、像素直径、遮挡、模糊、是否在球门/人群附近。val15人审帧只用于审计，不进入训练。

### 5.3 Teacher验收

按ball直径分桶报告 precision/recall：

- `<4 px`；
- `4-8 px`；
- `8-16 px`；
- `>16 px`。

进入pseudo生产的最低条件：

- 整体precision建议 >= 0.95；
- `>=8 px` ball recall建议 >= 0.85；
- 事件附近recall相对旧index有显著提升；
- 错误主要可由confidence、track或多尺度一致性过滤。

如果Teacher precision不足，不进行大规模伪标签训练；先调teacher阈值/tiling。如果Teacher recall不足但precision高，只将检出作为positive，未检出保持unknown，仍可做小规模pilot。

## 6. Phase 1：Pseudo Detection 生产

### 6.1 帧采样

不要对重叠10s窗口重复推理。先在视频绝对时间轴生成唯一帧清单：

- accepted shot/save附近 ±5秒：4-6 fps；
- accepted set_piece context与附近：2-4 fps；
- near-event context：2 fps；
- 普通背景：0.5-1 fps；
- 可选：当前模型高分但远离GT的片段只进入人工审计池，未经确认不作为负样本。

首轮pilot：20个train视频，要求覆盖不同场地、光照、运动模糊和摄像机运动。pilot通过后再扩展完整train。

### 6.2 多尺度与二次检测

- full-frame高分辨率主推理；
- full-frame无球结果时，可在球门邻域和人群密集区域做tile/secondary detection；
- 保留top-k候选及soft confidence，不强制唯一候选；
- 使用相邻帧track补全短时漏检，但插值框必须标记为`track_recovered`并降低监督权重；
- 不允许把渲染后的marker图像送给Teacher产生自证伪标签。

### 6.3 三态监督语义

每个帧/空间位置的监督状态：

1. `positive`：高置信teacher、人工GT或可靠track一致候选；
2. `unknown`：无检出、低置信、候选冲突、严重模糊；
3. `trusted_negative`：仅限人工确认无球，或多尺度/多teacher一致确认且画面质量足够。

第一版允许没有trusted-negative帧。positive帧内部远离目标的patch可作为有限空间背景，但必须避免把第二候选球附近强制为负。

### 6.4 质量过滤与权重

建议监督权重：

```text
human GT                   1.00
high-conf multi-scale      0.80-1.00
high-conf single-scale     0.60-0.80
track-recovered            0.30-0.60
low-conf / conflict        0.00 (unknown)
missing detection          0.00 (unknown)
```

具体阈值由Phase 0 PR曲线确定，不预先拍脑袋固定。

## 7. Phase 2：DINO Ball-awareness 表征实验

### 7.1 空间分辨率限制

当前512x896、patch size16对应约32x56 patch grid。许多足球小于一个patch，因此首版不训练精确box regression，而训练区域级soft heatmap/query：

- bbox中心映射至patch grid；
- 目标为最小半径1至2个patch的Gaussian；
- sigma可随ball像素尺寸变化，但有最小值；
- 多候选使用加权max或soft union；
- loss仅在有效监督mask内计算。

### 7.2 Ball head

首版建议：

```text
multi-layer DINO patch tokens
  -> layer norm + small projection
  -> learned ball query / 1x1 patch classifier
  -> ball heatmap [T,Hpatch,Wpatch]
  -> confidence-aware weighted pooling
  -> ball token [T,D]
  -> ball validity/presence [T]
```

不使用CLS直接预测球位置。Ball head必须从事件模型实际使用的patch层读取，推荐先复用当前 `[-8,-4,-1]` 中较浅/中层与最后层的轻量融合。

### 7.3 可选高分辨率cross-view蒸馏

如果 `<8 px` ball在全图patch上不可学，第二阶段再加入：

```text
原始HQ球周围context crop -> teacher/local encoder embedding
全图对应patch/query        -> student embedding
                         -> contrastive/distillation loss
```

它用于突破亚patch信息瓶颈，但不进入首轮B1，避免同时引入局部编码计算变量。

### 7.4 表征训练参数建议

首版采用低风险适配：

- backbone：DINOv3-L；
- 初始化：后续由同口径长视频结果选定的 last6/r8 或 last8/r8，不预设last8更优；
- LoRA：先只训练最后2-4 blocks中的QKV/attn proj，rank8；
- ball head：全训练；
- temporal/event heads：Phase 2可冻结或仅做低LR联合保护；
- detection loss占总有效梯度初始约10%-20%；
- 保留事件clip/dense loss或原始DINO feature consistency，避免只为找球破坏全局足球语义。

### 7.5 Loss

```text
L_total = L_event
        + lambda_ball * L_ball_heatmap
        + lambda_presence * L_ball_presence_masked
        + lambda_feature * L_feature_preserve
        + optional lambda_distill * L_cross_view_distill
```

- `L_ball_heatmap`：confidence-weighted focal BCE或soft BCE；
- `L_ball_presence_masked`：只在positive/trusted-negative帧计算；
- `L_feature_preserve`：适配前后非球区域或全局CLS的teacher-student一致性；
- 所有loss按有效监督数量归一化，不能因一个batch伪标签少而尺度跳变；
- 自动loss balancing只在固定权重pilot稳定后考虑，首轮保持可解释性。

## 8. Phase 3：事件模型 Feature Routing

### 8.1 不能只做辅助任务

ball heatmap指标提升不等于事件理解提升。Ball token必须进入每类事件的state representation：

```text
S_t^c = MLP_c([
    global_frame_t,
    existing_class_patch_evidence_t^c,
    ball_token_t,
    ball_validity_t,
    optional goal/player evidence_t
])
```

随后 `S_t^c` 进入 Temporal Stem/Encoder，再产生 dense curve和clip logits。

### 8.2 类别差异化

- shot：ball token、ball temporal displacement、goal relation为强证据，但普通传球/回传必须作为ball-present negative context；
- save：使用ball token + shot history/response + goal/keeper-region context；不能只复制shot分数；
- set_piece：ball是可选证据。准备阶段可能看不到球，必须允许global/player-layout/context主导，避免ball漏检损害recall。

### 8.3 缺失保护

- detector/predicted ball缺失时使用`unknown/no-observation token`；
- 不使用“零ball token=没有球”的语义；
- 训练时进行ball evidence dropout、坐标jitter、confidence noise；
- global feature始终保留，不能让ball branch成为硬门控；
- 评测必须单独报告teacher-hit、teacher-missing和small-ball子集。

## 9. 对照实验矩阵

所有实验使用相同train/val/external split、相同seed、相同输入、相同epoch预算和长视频协议。

### B0：固定baseline

- 当前选定的最优事件checkpoint；
- 无ball loss、无ball token；
- 重新缓存val15完整dense输出作为锚点。

### B1：Detection auxiliary only

- 增加ball heatmap head和loss；
- ball token不进入Temporal Encoder。

目的：验证DINO是否学到ball awareness，以及“只加辅助loss”是否足够。

### B2：Ball token routing

- 在B1基础上将ball token进入class-conditioned state；
- 保持其他结构不变。

目的：验证事件模型是否实际利用ball evidence。B2是核心实验。

### B3：External GT only

- 只用外部真实检测GT训练ball awareness；
- 不使用当前域pseudo。

### B4：Pseudo only

- 只用当前域高质量pseudo positives；

### B5：External GT + pseudo

- B3与B4组合；
- 仅在pilot资源允许时运行。

优先顺序：B0 -> B1 -> B2。B3/B4/B5用于判断数据来源价值，不应在首轮全部并发烧卡。

## 10. 评测协议

### 10.1 Ball representation指标

- heatmap hit@1 patch、hit@2 patches；
- pixel-size分桶recall；
- held-out video ball linear probe AUROC/AUPRC；
- positive patch与草皮线、球鞋、广告牌等易混淆patch的feature gap；
- teacher-hit/missing/track-recovered分组；
- ball token temporal continuity与抖动。

### 10.2 因果利用审计

在同一批事件clip上执行：

- 正常ball token；
- mask ball token；
- 随机打乱时间顺序的ball token；
- 将ball token换成其他视频。

如果事件logits与指标几乎不变，说明routing未被使用；如果所有类别都大幅下降，说明模型过度依赖ball shortcut。

### 10.3 下游长视频指标

主协议保持：val15定阈值，external18 untouched；PointNMS，1:1匹配，tol=5s。

报告：

- per-class `Precision @ Recall >= 85%`；
- per-class `Precision @ Recall >= 90%`或最大recall ceiling；
- event-level AP；
- FP/90min；
- per-video AP和worst-20% eligible-video AP；
- per-video recall P10/median；
- 去重后的人工参与时长；
- full-support与capped10/15/20 coverage recall；
- 视频级bootstrap置信区间。

sampled clip-val F1只作为diagnostic。

## 11. 预注册验收标准

### Phase 2通过条件

- held-out video ball hit@2 patches相对冻结DINO probe有显著提升；
- ball feature gap提高并超过视频级bootstrap CI；
- 非球区域/全局特征没有明显整体坍塌；
- detection有效监督帧比例足够，loss不是由极少数帧驱动。

### Phase 3通过条件

相对B0：

- shot或save在同recall下precision提高至少2个百分点；
- candidate recall ceiling下降不超过1个百分点；
- FP/90min或人工参与时长下降至少10%；
- teacher-missing/small-ball子集召回不明显下降；
- 改善超过视频级bootstrap CI。

set_piece不要求一定依赖ball，但不得因ball branch降低recall超过1个百分点。

## 12. 失败判定与决策树

1. Teacher precision不足：停止pseudo生产，先修teacher或增加人工GT。
2. Teacher precision高但recall低：只用positive/unknown训练，不造负样本；pilot仍可继续。
3. B1 heatmap不提升：当前全图patch分辨率不足或LoRA层选择不对；优先试cross-view高分辨率蒸馏，不加大loss。
4. B1 heatmap提升、B2事件不提升：routing/时序利用失败；检查causal mask审计，不继续扩大检测数据。
5. B2只在teacher-hit子集提升、missing子集下降：过度依赖ball，增加evidence dropout和unknown token训练。
6. shot提升、save不提升：引入shot-history + ball/goal条件化，而不是继续加ball loss。
7. sampled-val提升、长视频不提升：Train-online mismatch仍在；将已验证的Online-simulation协议与B2组合，但单独做新实验。
8. external18不转移：当前域pseudo过拟合场地/视频；提高视频均衡和外部GT比例，审计domain shift。

## 13. 资源与时间估算方法

Teacher生产时间：

```text
总时长 = 唯一采样帧数 * 单帧Teacher耗时 / 并行GPU数
```

启动前必须实测三组batch/分辨率，不以宣传FPS估算。pilot先限制20视频；全量扩展只在B1表征指标正向后进行。

Ball head本身计算很小；主要额外开销来自：

- 多层patch token保留；
- 可能的高分辨率cross-view；
- pseudo index读取。

第一版B1/B2不增加额外视频解码和DINO forward，只在现有patch tokens上增加轻量head/routing，目标训练开销增幅控制在10%以内。

## 14. 工程交付物

### 数据

- `ball_teacher_audit/frames.jsonl`
- `ball_teacher_audit/manual_review.csv`
- `ball_teacher_audit/summary.json`
- 版本化pseudo index目录；
- train/val/external视频覆盖清单与泄漏检查报告。

### 代码

- 强Teacher离线推理与分片恢复脚本；
- normalized-coordinate pseudo index reader；
- confidence/track-aware heatmap target builder；
- BallHeatmapQueryHead；
- class-conditioned ball routing；
- causal token ablation评测脚本；
- 多卡Teacher watchdog和断点续跑。

### 日志

每个epoch记录：

- effective positive/unknown/trusted-negative帧数；
- teacher source和confidence分布；
- ball heatmap loss/hit@patch；
- ball token validity和dropout率；
- 每类事件loss与gradient norm；
- val15事件指标和ball probe指标。

## 15. 启动前待确认参数

以下信息由Phase 0自动探测或由用户补充，不应猜测：

1. 强慢Teacher的代码入口和checkpoint路径；
2. Teacher推荐输入分辨率、是否支持batch/tile；
3. 单帧/单视频实际推理速度；
4. 外部足球检测数据路径、标签格式、视频数量、license；
5. 是否包含球门/守门员标注；
6. 当前事件train视频中可用于pseudo生产的最终HQ路径优先级。

在这些参数确认后，先提交Phase 0审计命令和预计完成时间，再启动任何全量任务。
