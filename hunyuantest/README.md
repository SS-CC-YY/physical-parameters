# HunyuanVideo-1.5 Test Runner

目标：用你已经通过 ModelScope 下载的 `Tencent-Hunyuan/HunyuanVideo-1.5` 做第一阶段可行性测试，先确认模型能在服务器 GPU 7 上生成视频，再用简单的球心跟踪脚本检查生成视频里是否能拟合出物理量 proxy。

这个目录不复制 HunyuanVideo-1.5 官方源码。官方模型需要配合官方代码仓库的 `generate.py` 使用；本目录提供的是 benchmark 侧的 wrapper、manifest 和评估脚本。

参考官方信息：

- HunyuanVideo-1.5 官方仓库要求 Linux、Python 3.10+、CUDA/PyTorch 兼容环境，并建议开启 offloading 时至少 14GB 显存。
- 官方推理入口是 `generate.py`，核心参数包括 `--model_path`、`--prompt`、`--resolution`、`--image_path`、`--output_path`。
- `--image_path none` 是 T2V；传入图片路径就是 I2V。

## Files

- `run_hunyuan15_official.py`：批量调用官方 `generate.py`，支持 I2V/T2V manifest、GPU 选择、日志、metadata。
- `build_i2v_manifest.py`：从 `blender/seeds` 生成 HunyuanVideo-1.5 I2V 测试 manifest。
- `evaluate_freefall.py`：对生成视频做橙色球心跟踪，并拟合 `y(t) = a t^2 + b t + c`，输出 `g_proxy_px_per_s2 = 2a`。
- `requirements.txt`：本目录脚本需要的最小依赖，不包含官方 HunyuanVideo-1.5 推理依赖。
- `manifests/v1a_cam_side_explicit_g_hunyuan15.jsonl`：默认 4 个 v1_A/CAM_Side I2V 测试任务。

## 1. Server Layout

建议服务器目录保持为：

```text
~/data/heyuanyu/yefei/chenyu/data/
  physical-parameters/          # 本 repo
  HunyuanVideo-1.5-code/         # 官方源码，里面要有 generate.py
  models/HunyuanVideo-1.5/       # ModelScope 下载的权重目录
```

如果你之前没有指定 `--local_dir`，先定位 ModelScope 实际下载路径：

```bash
find ~/.cache/modelscope -maxdepth 6 -type d -iname "*HunyuanVideo*1*5*" 2>/dev/null
```

更推荐重新指定清晰路径，避免后续命令找错：

```bash
mkdir -p ~/data/heyuanyu/yefei/chenyu/data/models
modelscope download \
  --model Tencent-Hunyuan/HunyuanVideo-1.5 \
  --local_dir ~/data/heyuanyu/yefei/chenyu/data/models/HunyuanVideo-1.5
```

然后拉官方代码：

```bash
cd ~/data/heyuanyu/yefei/chenyu/data
git clone --depth 1 https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git HunyuanVideo-1.5-code
```

## 2. Install Dependencies

先使用已有的可用 PyTorch 环境。你之前服务器的 `base` 里已有 `torch 2.5.1+cu124`，优先复用它，不要在一个空 venv 里重新装 torch。

安装本测试目录脚本依赖：

```bash
cd ~/data/heyuanyu/yefei/chenyu/data/physical-parameters
python -m pip install -r hunyuantest/requirements.txt
```

安装官方 HunyuanVideo-1.5 推理依赖：

```bash
cd ~/data/heyuanyu/yefei/chenyu/data/HunyuanVideo-1.5-code
python -m pip install -r requirements.txt
python -m pip install -i https://mirrors.tencent.com/pypi/simple/ --upgrade tencentcloud-sdk-python
```

如果 `requirements.txt` 又开始下载大量 `nvidia-*` CUDA wheel，说明当前环境没有可复用的 torch，先中断，回到已有 torch 的 `base` 环境或使用 `python -m venv --system-site-packages` 继承 base。

## 3. Run One I2V Smoke Test On GPU 7

```bash
cd ~/data/heyuanyu/yefei/chenyu/data/physical-parameters

python hunyuantest/run_hunyuan15_official.py \
  --manifest hunyuantest/manifests/v1a_cam_side_explicit_g_hunyuan15.jsonl \
  --hunyuan-repo ../HunyuanVideo-1.5-code \
  --model-path ../models/HunyuanVideo-1.5 \
  --outdir outputs/hunyuan15_i2v_v1a_smoke \
  --gpu-id 7 \
  --resolution 480p \
  --max-jobs 1 \
  --rewrite false \
  --sr false \
  --offloading true \
  --overlap-group-offloading false
```

输出目录：

```text
outputs/hunyuan15_i2v_v1a_smoke/
  videos/*.mp4
  metadata/*.json
  logs/*.stdout.log
  logs/*.stderr.log
  generation_log.jsonl
```

## 4. Run The 4-Job Feasibility Set

```bash
python hunyuantest/run_hunyuan15_official.py \
  --manifest hunyuantest/manifests/v1a_cam_side_explicit_g_hunyuan15.jsonl \
  --hunyuan-repo ../HunyuanVideo-1.5-code \
  --model-path ../models/HunyuanVideo-1.5 \
  --outdir outputs/hunyuan15_i2v_v1a \
  --gpu-id 7 \
  --resolution 480p \
  --rewrite false \
  --sr false \
  --offloading true \
  --overlap-group-offloading false
```

如果你确认下载的是 480p I2V step-distilled 权重，并且官方模型目录能自动识别对应子模型，可以测试 8 步快速推理：

```bash
python hunyuantest/run_hunyuan15_official.py \
  --manifest hunyuantest/manifests/v1a_cam_side_explicit_g_hunyuan15.jsonl \
  --hunyuan-repo ../HunyuanVideo-1.5-code \
  --model-path ../models/HunyuanVideo-1.5 \
  --outdir outputs/hunyuan15_i2v_v1a_step_distill \
  --gpu-id 7 \
  --resolution 480p \
  --num-inference-steps 8 \
  --enable-step-distill true \
  --cfg-distilled true \
  --rewrite false \
  --sr false \
  --offloading true \
  --overlap-group-offloading false
```

## 5. Evaluate Generated Videos

```bash
python hunyuantest/evaluate_freefall.py \
  --videos-dir outputs/hunyuan15_i2v_v1a/videos \
  --metadata-dir outputs/hunyuan15_i2v_v1a/metadata \
  --outdir outputs/hunyuan15_i2v_v1a/eval \
  --fps 24
```

主要看：

```text
outputs/hunyuan15_i2v_v1a/eval/summary.csv
outputs/hunyuan15_i2v_v1a/eval/tracks/*.csv
```

`g_proxy_px_per_s2` 是像素坐标下的加速度 proxy，不是物理单位 m/s^2。第一阶段只看它是否能随 prompt 里的 `g_hidden` 单调变化。

## 6. Build A New Manifest

例如生成 `v1_A` 四个重力变体、`CAM_Side` 的 I2V manifest：

```bash
python hunyuantest/build_i2v_manifest.py \
  --seeds-root blender/seeds \
  --version v1 \
  --experiments v1_A \
  --cameras CAM_Side \
  --output hunyuantest/manifests/v1a_cam_side_explicit_g_hunyuan15.jsonl
```

也可以限制任务数：

```bash
python hunyuantest/build_i2v_manifest.py \
  --seeds-root blender/seeds \
  --version v1 \
  --experiments v1_A \
  --cameras CAM_Side \
  --max-jobs 1 \
  --output hunyuantest/manifests/smoke_one_job.jsonl
```

## 7. Common Errors

`generate.py not found`

`--hunyuan-repo` 必须指向官方代码目录，不是模型权重目录。该目录下应存在：

```text
HunyuanVideo-1.5-code/generate.py
```

`model path not found`

`--model-path` 必须指向 ModelScope 下载出来的权重目录。建议使用 `--local_dir` 固定路径。

`CUDA out of memory`

先保守运行：

```bash
--resolution 480p --sr false --offloading true --overlap-group-offloading false --max-jobs 1
```

`torchrun: command not found`

说明当前 Python 环境不完整。先确认：

```bash
python -m torch.distributed.run --help
```

如果这个命令存在，本 wrapper 会自动用它替代 `torchrun`。
