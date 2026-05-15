# baselineV1 实验对照表

## 设计目标

baselineV1 面向“单物体、单运动、单隐藏物理参数”的最简物理 probe。所有实验都尽量围绕同一种小球展开，便于控制视觉因素，并直接从轨迹中反推单个隐式物理量。

## 对照表

| 实验 | 对应文件 | 运动描述 | 隐式物理量 | 关键公式 | 推荐拟合反推方法 |
| --- | --- | --- | --- | --- | --- |
| A | `baselinev1_A_ball_freefall_gravity.blend` / `exp_A_ball_freefall_gravity.py` | 小球自由落体直到接触地面 | 重力加速度 `g_hidden` | `z(t)=z0-\frac{1}{2}gt^2` | 从球心竖直轨迹 `z(t)` 做二次曲线拟合，二次项直接反推 `g` |
| B | `baselinev1_B_ball_bounce_restitution.blend` / `exp_B_ball_bounce_restitution.py` | 小球竖直下落并重复弹跳 | 恢复系数 `e_hidden` | `v_{after}=e|v_{before}|` | 用连续两次落地前后的速度比，或相邻峰值高度比 `h_{k+1}/h_k=e^2` 反推 `e` |
| C | `baselinev1_C_ball_slide_friction.blend` / `exp_C_ball_slide_friction.py` | 小球在水平地面上减速滑行直到停止 | 摩擦系数 `mu_hidden` | `x(t)=x0+v0t-\frac{1}{2}\mu g t^2` | 对水平位移 `x(t)` 或速度 `v(t)` 做匀减速拟合，减速度除以已知 `g` 得到 `\mu` |
| D | `baselinev1_D_ball_pendulum_damping.blend` / `exp_D_ball_pendulum_damping.py` | 小球做阻尼摆动 | 阻尼系数 `gamma_hidden` | `\theta(t)=\theta_0 e^{-\gamma t}\cos(\omega_d t)` | 提取摆角或球心横向位移包络，对指数衰减项做拟合得到 `\gamma` |

## 建议用法

- A 适合作为最纯净的重力估计基线。
- B 适合测试模型是否能从弹跳节律中恢复接触参数。
- C 适合测试模型对持续减速运动的理解。
- D 适合测试模型对周期运动与阻尼衰减的区分能力。
