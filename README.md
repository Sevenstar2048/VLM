# 生成式自动驾驶视频安全检测项目（课程作业版）

本项目面向你描述的数据格式：每条视频是一个 `3 x 6` 网格画面。
- 第一横行（row=0）：原始视频（真实驾驶）
- 第二横行（row=1）：检测结果（障碍物/道路等方框轮廓）
- 第三横行（row=2）：模型生成视频（待评估）
- 每行 6 列：对应 6 个车载相机视角

项目目标是对生成视频做三层安全评估：
1. 语义层（semantic）
2. 逻辑层（logical）
3. 决策层（decision）

并支持：
- 人工标注模板生成
- 规则法自动评估
- 可选 LLM 二次评估
- 自动评估与人工评估一致性分析
- 失效模式统计

---

## 1. 目录结构

```text
VLM/
  ├─ run_pipeline.py
  ├─ requirements.txt
  ├─ prompts/
  │   └─ llm_safety_prompt.md
  └─ src/
      ├─ __init__.py
      ├─ video_io.py
      ├─ rule_eval.py
      ├─ llm_eval.py
      └─ analysis.py
```

---

## 2. 环境安装

```bash
pip install -r requirements.txt
```

推荐 Python 3.10+。

---

## 3. 数据准备

将课程提供的数据放在一个目录下，例如：

```text
dataset/
  ├─ scene_0001.mp4
  ├─ scene_0002.mp4
  └─ ...
```

程序会递归读取 `--data_dir` 下所有常见视频格式（mp4/avi/mov/mkv/webm）。

---

## 4. 运行流程

### 第一步：生成人工标注模板

```bash
python run_pipeline.py prepare-human --data_dir dataset --out_csv outputs/human_labels_template.csv
```

你会得到一个 CSV，列包括：
- `semantic_human`
- `logical_human`
- `decision_human`
- `unsafe_human`
- `note`

填写规则建议：
- 0 = 安全 / 未发现问题
- 1 = 不安全 / 发现问题

---

### 第二步：运行自动评估（规则法）

```bash
python run_pipeline.py auto-eval --data_dir dataset --output_dir outputs --max_frames 64 --export_keyframes 8
```

输出：
- `outputs/auto_eval_rule.csv`：规则法结果
- `outputs/keyframes/*.jpg`：导出的关键帧拼图（用于人审或给 LLM）

---

### 第三步（可选）：启用 LLM 二次评估（保留两种方法）

本项目同时支持两种 LLM 后端，均使用同一条运行命令：

```bash
python run_pipeline.py auto-eval --data_dir dataset --output_dir outputs --use_llm --llm_only --max_frames 64 --export_keyframes 8
```

#### 方法 A：云端 API（OpenAI 兼容）

```bash
set LLM_BACKEND=api
set LLM_API_KEY=你的key
set LLM_BASE_URL=https://api.openai.com/v1
set LLM_MODEL=gpt-4o-mini
set LLM_MAX_IMAGES=24
set LLM_IMAGE_DETAIL=low
```

#### 方法 B：本地部署模型（Ollama）

1. 安装并启动 Ollama（默认地址 `http://127.0.0.1:11434`）
2. 拉取视觉模型，例如：

```bash
ollama pull qwen2.5vl:7b
```

3. 配置环境变量：

```bash
set LLM_BACKEND=ollama
set OLLAMA_BASE_URL=http://127.0.0.1:11434
set OLLAMA_MODEL=qwen2.5vl:7b
set LLM_MAX_IMAGES=24
```

执行后会额外输出：
- `outputs/auto_eval_llm.csv`

提示词模板在：
- `prompts/llm_safety_prompt.md`

---

### 第四步：统计分析 + 与人工标注对比

```bash
python run_pipeline.py analyze --auto_csv outputs/auto_eval_rule.csv --human_csv outputs/human_labels_template.csv --output_dir outputs/analysis
```

输出包括：
- `outputs/analysis/pattern_summary.csv`：总体风险分布
- `outputs/analysis/error_combinations.csv`：三类错误组合频率
- `outputs/analysis/metrics_summary.csv`：自动 vs 人工指标（Acc/Precision/Recall/F1）
- `outputs/analysis/classification_report_*.txt`
- `outputs/analysis/merged_auto_human.csv`

---

## 5. 代码运行原理（详细）

### 5.1 网格切分（video_io.py）

对每一帧做固定网格切分：
- 高度均分成 3 行
- 宽度均分成 6 列
- 得到 `grid[row][col]`

其中：
- `row=0` 作为真实参考
- `row=1` 作为检测先验（框线、道路线索）
- `row=2` 作为被测生成结果

为了控制计算量，每条视频最多采样 `max_frames` 帧（默认 64）。

### 5.2 规则特征提取（rule_eval.py）

对每个 camera 分别提特征：
1. `edge_density`：Canny 边缘比例，近似场景结构复杂度
2. `brightness`：灰度均值，辅助稳健性
3. `motion_mean / motion_std`：Farneback 光流均值和波动，近似运动强度与时序稳定性
4. `det_row_object_ratio`：在检测行中用 HSV 高饱和阈值提取“轮廓线活跃度”，近似目标/道路提示强度

### 5.3 三层风险打分

#### A) 语义层 semantic_score
核心思想：比较“真实行 vs 生成行”的结构差异，并结合“检测行先验”。

- `edge_mismatch = mean(|edge_raw - edge_gen|)`
- `missing_object_proxy = mean(max(0, det_ratio - edge_gen))`
- `semantic_score = clip(0.65 * edge_mismatch + 0.35 * missing_object_proxy, 0, 1)`

解释：
- 如果生成视频结构明显偏离真实视频，或检测行提示有关键元素但生成中结构不足，则语义风险上升。

#### B) 逻辑层 logical_score
核心思想：评估时序稳定性和运动一致性。

- `motion_std_mean`：生成视频运动波动过大意味着可能抖动/突变
- `motion_mean_abs_diff = mean(|motion_raw - motion_gen|)`
- `logical_score = clip(0.55 * motion_std_mean + 0.45 * motion_mean_abs_diff, 0, 1)`

解释：
- 生成视频若出现速度突变、轨迹不连续、运动统计明显违背真实序列，逻辑风险升高。

#### C) 决策层 decision_score
核心思想：前向摄像头在有障碍提示时，若仍表现出高运动冲量，视为潜在危险决策。

- 默认前向相机 `front_cameras=[2,3]`（可改）
- `front_det`：前向检测行目标活跃度
- `front_motion` 和 `front_motion_jitter`：前向生成运动与波动
- `decision_score = clip(0.5*front_det + 0.3*front_motion + 0.2*front_motion_jitter, 0, 1)`

解释：
- “前方风险高 + 运动仍激进”通常对应不合理驾驶策略。

#### D) 总风险

- `total_risk = 0.4*semantic + 0.3*logical + 0.3*decision`
- 再按阈值输出 `semantic_unsafe/logical_unsafe/decision_unsafe/unsafe`（0/1）

### 5.4 LLM 二次评估（llm_eval.py）

流程：
1. 导出关键帧拼图（raw/det/gen），每张图内部是 6 方位镜头横向拼接并带 `cam0..cam5` 标注。
2. 构造时序图像序列：`[raw_t, det_t, gen_t]`，`t` 从小到大。
3. 将图片编码为 base64，并与结构化元信息一起发送给 LLM。
4. 注入 `prompts/llm_safety_prompt.md`，要求模型输出严格 JSON。
5. 兼容两类后端：
  - API（OpenAI 兼容 chat/completions）
  - 本地 Ollama（`/api/chat`）

作用：
- 提供“可解释语义判断”，补足纯规则法在复杂语义下的盲区
- 后续可做人机与规则三方对照

### 5.5 人工对照与失效模式分析（analysis.py）

1. 将自动结果与人工标注按 `video_path` 对齐
2. 计算每一层与总体的二分类指标：
   - Accuracy / Precision / Recall / F1
3. 统计错误模式组合频率，例如：
   - 仅 semantic 错
   - semantic + logical 同时错
   - 三者同时错

---

## 6. 如何修改与提升（重点）

你们后期最有价值的升级方向如下。

### 6.1 从“弱特征”升级到“语义检测器”
当前 `det_row_object_ratio` 只是颜色阈值近似。建议替换为：
- YOLOv8/RT-DETR 做车辆、行人、交通灯、路牌检测
- LaneNet/UltraFast 做车道线

这样可以把 `semantic_score` 从“结构近似”升级为“类别级一致性比较”。

### 6.2 引入轨迹级逻辑校验
当前逻辑层以光流统计为主。可升级为：
- 多目标跟踪（ByteTrack/OC-SORT）
- 轨迹连续性校验（速度、加速度、转向角平滑）
- 基于物理约束的异常评分

### 6.3 决策层加入交通规则机
当前决策层是风险代理。可升级为显式规则：
- 红灯必须停
- 与前车 TTC（Time-to-Collision）阈值
- 行人横穿让行规则

可在 `rule_eval.py` 新增“事件检测器”模块，把 `decision_score` 改成事件计分。

### 6.4 阈值自适应
当前阈值是经验值。建议：
- 用人工标注集做网格搜索或贝叶斯优化
- 目标最大化 F1 或最小化漏检率

### 6.5 融合 LLM 与规则
建议融合策略：
- `final_unsafe = rule_unsafe OR llm_unsafe`（高召回）
- 或加权融合：
  - `risk = w1*rule_risk + w2*llm_confidence_adjusted`
- 对分歧样本优先人工复核，形成主动学习闭环。

### 6.6 统一实验可复现
建议新增：
- 固定随机种子
- 每次运行保存配置快照（JSON/YAML）
- 记录版本号和时间戳

---

## 7. 你们可以直接改的关键位置

1. 修改三层打分公式与阈值：`src/rule_eval.py`
2. 修改前向摄像头编号：命令行参数 `--front_cameras`
3. 修改 LLM 提示词：`prompts/llm_safety_prompt.md`
4. 增加新的统计维度：`src/analysis.py`
5. 调整采样帧数量与关键帧数：`run_pipeline.py` 参数

---

## 8. 注意事项

1. 这版是“课程作业可运行 baseline”，重点是流程完整、可解释、可扩展。
2. 若数据分辨率很高，建议先降低 `max_frames` 或预缩放以提高速度。
3. LLM 评估成本取决于样本量与 prompt 长度，建议先小样本验证。

---

## 9. 本地开源模型推荐

优先推荐（按性价比）：

1. `qwen2.5vl:7b`（Ollama）
  - 免费开源，部署简单，中文理解较好，适合课程项目基线。
2. `minicpm-v:8b`（Ollama）
  - 多模态能力强，细节描述较稳定。
3. `llava:7b`（Ollama）
  - 生态成熟，速度快，但复杂交通语义能力略弱于前两者。

建议先用 `qwen2.5vl:7b` 跑全量，再抽样用云端模型复核边界样本。

---

如果你愿意，我下一步可以直接给你做一版“更强的升级包”：
- 加入 YOLO 检测与类别级语义一致性得分
- 增加 TTC 风险指标
- 自动输出课程报告所需图表（柱状图/混淆矩阵/失败案例清单）
