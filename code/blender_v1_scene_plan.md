# Blender 室内场景下的 Baseline V1 组织方案

## 1. 顶层集合建议
- `ENV_Room`：原有室内环境（墙、窗、柜、桌、灯）
- `CAM_Main`：相机与相机辅助物
- `LGT_Main`：主灯、补光、环境光辅助
- `EXP`：所有实验集合的父集合
  - `EXP_FREEFALL`
  - `EXP_PROJECTILE`
  - `EXP_SPRING`
  - `EXP_PENDULUM`
  - `EXP_ROTATION`
- `AUX_Render`：可选的遮挡板、mask辅助物、标定参考物

## 2. 命名规则建议
### 环境
- `ENV_*`：静态环境物体
- `CAM_*`：相机相关
- `LGT_*`：灯光相关

### 实验通用
- `EXP_ROOT`：实验总原点
- `EXP_Platform`：桌面上的统一实验平台
- `MAT_*`：材质

### 自由落体
- `FF_GlassTube`
- `FF_TopCap`
- `FF_BottomBase`
- `FF_Ball`

### 平抛/斜抛
- `PR_GlassBox`
- `PR_Launcher`
- `PR_GuideRail`
- `PR_Floor`
- `PR_Ball`

### 弹簧振子
- `SP_Stand`
- `SP_Rail`
- `SP_Slider`
- `SP_SpringVis`

### 单摆
- `PD_Post`
- `PD_Arm`
- `PD_Pivot`
- `PD_Line`
- `PD_Bob`

### 自转
- `RT_Base`
- `RT_Disc`
- `RT_Marker`

## 3. 每个实验的对象清单
### 3.1 自由落体（真空仓）
必需：透明管、顶部释放盖、底部基座、球体
推荐：小铭牌、固定夹具、缓冲底座

### 3.2 平抛/斜抛（透明实验箱）
必需：透明箱体、发射器、导轨、底板、球体
推荐：发射器底座、球回收槽、侧边支撑

### 3.3 弹簧振子（水平导轨）
必需：立柱、导轨、滑块、弹簧可视化件
推荐：弹簧固定扣、限位块、刻度铭牌

### 3.4 小角度单摆
必需：支柱、横臂、支点、摆线、摆球
推荐：支点夹具、底座、背景挡板

### 3.5 匀速/匀加速自转
必需：底座、电机壳体样式的转台、圆盘、方向标记
推荐：轴心盖、外壳、控制盒

## 4. 统一渲染与拍摄建议
- 相机固定，不同实验尽量共用同一主相机
- 焦距中等偏长，减少透视畸变
- 关闭或减弱运动模糊，便于轨迹拟合
- 背景尽量统一，避免每个实验更换太多装饰
- 光照稳定，不要做强烈彩色灯

## 5. 数据输出建议
每个实验至少输出：
- `*_trajectory.csv`：逐帧位置/角度真值
- `*_params.json`：实验参数
- `*.mp4`：RGB视频

推荐额外输出：
- 二值mask序列
- 深度图或法线图
- 相机内参/外参JSON

## 6. 复用脚本用法
1. 在 Blender 打开室内 `.blend`
2. 打开脚本编辑器，载入 `blender_v1_template.py`
3. 修改顶部 `CFG["experiment"]`
4. 修改对应实验参数
5. 运行脚本
6. 在 `//outputs/` 查看 CSV / JSON / 视频
