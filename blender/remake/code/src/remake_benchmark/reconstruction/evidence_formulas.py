"""Human-readable inverse-physics formulas used by the evidence report.

This module is deliberately independent from the numerical fitters.  It is a
presentation registry: the equations and steps below describe what
``physics_parameters.py`` already computes, but importing this module cannot
run a fit or access a target parameter.  Report builders can therefore render
the same concise explanation for every one of the 13 experiments and 24
experiment-parameter response axes.

The diagnostic helpers are also intentionally pure.  They walk an existing
``fit["diagnostics"]`` value and return new JSON-serializable objects without
modifying the supplied fit result.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "1.1.0"


# This is the frozen experiment-system order used by the benchmark and its
# reports.  A system may expose more than one independently scanned parameter
# channel; it must still appear exactly once in an experiment-level figure.
EXPERIMENT_ORDER: tuple[str, ...] = (
    "v1_A",
    "v1_B",
    "v1_C",
    "v1_D",
    "v2_A",
    "v2_B",
    "v2_C",
    "v2_D",
    "v2_E",
    "v3_A",
    "v3_B",
    "v3_C",
    "v3_D",
)


def _parameter(
    title_zh: str,
    title_en: str,
    symbol: str,
    unit: str,
    estimate_plain: str,
    estimate_latex: str,
    steps_zh: Sequence[str],
) -> dict[str, Any]:
    return {
        "parameter_title_zh": title_zh,
        "parameter_title_en": title_en,
        "symbol": symbol,
        "unit": unit,
        "estimate_plain": estimate_plain,
        "estimate_latex": estimate_latex,
        "estimate_steps_zh": list(steps_zh),
    }


def _experiment(
    title_zh: str,
    title_en: str,
    observables: Sequence[str],
    model_plain: str,
    model_latex: str,
    fitter_function: str,
    fitter_method: str,
    fitter_summary_zh: str,
    parameters: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "experiment_title_zh": title_zh,
        "experiment_title_en": title_en,
        "observables": list(observables),
        "model_plain": model_plain,
        "model_latex": model_latex,
        "fitter_function": fitter_function,
        "fitter_method": fitter_method,
        "fitter_summary_zh": fitter_summary_zh,
        "parameters": {name: dict(spec) for name, spec in parameters.items()},
    }


# Keep the method strings synchronized with the ``method=`` values returned by
# the 13 fitters in physics_parameters.py.  Formula text is explanatory only;
# the numerical implementation remains the source of computed estimates.
FORMULA_REGISTRY: dict[str, dict[str, Any]] = {
    "v1_A": _experiment(
        "短时竖直自由落体",
        "Short-horizon vertical free fall",
        ("time_s", "z_m"),
        "z(tau) = c0 + c1*tau + c2*tau^2",
        r"z(\tau)=c_0+c_1\tau+c_2\tau^2",
        "_fit_v1a",
        "first_motion_to_first_contact_quadratic",
        "从首次持续向下运动截取到第一次地面接触，并只在这段轨迹上作稳健二次回归。",
        {
            "gravity_g": _parameter(
                "重力加速度",
                "Gravitational acceleration",
                "g",
                "m/s^2",
                "g_hat = -2*c2",
                r"\hat g=-2c_2",
                (
                    "先跳过释放前的静止前缀，定位首次持续向下运动和第一次地面接触。",
                    "仅使用“运动开始—首次接触”段，以该段首时刻为零点，对 z(tau) 作稳健二次拟合。",
                    "由二次项系数 c2 计算 g_hat=-2*c2；方向、残差、加速度稳定性和 g>0 是独立可信度门控。",
                ),
            )
        },
    ),
    "v1_B": _experiment(
        "单次墙面碰撞",
        "Single wall impact",
        ("time_s", "x_m"),
        "x(t) is piecewise linear around one velocity reversal",
        r"x(t)=\begin{cases}a_-+v_-t,&t<t_c\\a_++v_+t,&t\ge t_c\end{cases}",
        "_fit_v1b",
        "single_impact_segmented_velocity_ratio_with_consistency_gate",
        "定位已知墙面附近的单次显著反向，排除接触邻域后分别回归碰撞前后的近似匀速段。",
        {
            "restitution_e": _parameter(
                "恢复系数",
                "Coefficient of restitution",
                "e",
                "1",
                "e_hat = abs(v_post/v_pre)",
                r"\hat e=\left|v_{\mathrm{post}}/v_{\mathrm{pre}}\right|",
                (
                    "跳过起始静止段，在已知墙面附近定位唯一显著速度反向，并在接触点两侧保留保护间隔。",
                    "对保护间隔外的碰撞前、后轨迹分别作线性拟合，得到 v_pre>0 与 v_post<0。",
                    "用 e_hat=|v_post/v_pre| 反推恢复系数，并检查墙面位置、分段匀速性、接触停留、物理范围及局部/全局估计一致性。",
                ),
            )
        },
    ),
    "v1_C": _experiment(
        "水平动摩擦减速",
        "Horizontal kinetic-friction deceleration",
        ("time_s", "x_m"),
        "x(tau) = c0 + c1*tau + c2*tau^2; a = 2*c2",
        r"x(\tau)=c_0+c_1\tau+c_2\tau^2,\quad a=2c_2",
        "_fit_v1c",
        "first_motion_fixed_initial_velocity_pre_stop",
        "截取首次持续正向运动到首次静止尾段，以独立估计并固定的初速度拟合恒减速度。",
        {
            "kinetic_friction_mu": _parameter(
                "动摩擦系数",
                "Kinetic-friction coefficient",
                "mu_k",
                "1",
                "mu_hat = -a/g_known = -2*c2/g_known",
                r"\hat\mu_k=-a/g_{\mathrm{known}}=-2c_2/g_{\mathrm{known}}",
                (
                    "定位首次持续正向运动和首次静止尾段，只保留中间的第一段摩擦滑行。",
                    "由运动开始后的短区间稳健估计初速度，固定该速度后拟合 x-x0-v0*tau=0.5*a*tau^2。",
                    "由 mu_hat=-a/g_known 反解摩擦系数，并检查单向减速、恒加速度、残差及 mu>=0。",
                ),
            )
        },
    ),
    "v1_D": _experiment(
        "固定周期阻尼摆",
        "Fixed-period damped pendulum",
        ("time_s", "x_m", "z_m", "theta_rad"),
        "theta(tau) = exp(-beta*tau)*(A*cos(omega*tau)+B*sin(omega*tau)); omega=2*pi/T0",
        r"\theta(\tau)=e^{-\beta\tau}[A\cos(\omega\tau)+B\sin(\omega\tau)],\quad\omega=2\pi/T_0",
        "_fit_v1d",
        "first_complete_cycle_fixed_period_decay_search",
        "由已知枢轴换算摆角，只在首个完整周期内按固定周期搜索最小残差的衰减率。",
        {
            "amplitude_decay_beta": _parameter(
                "振幅衰减率",
                "Amplitude-decay rate",
                "beta",
                "1/s",
                "beta_hat = argmin_beta MSE(theta, theta_model(beta))",
                r"\hat\beta=\arg\min_{\beta}\operatorname{MSE}(\theta,\theta_{\mathrm{model}}(\beta))",
                (
                    "用球心与已知枢轴计算展开角 theta，并用同相位极值定位完整周期。",
                    "只选择第一个完整周期，固定 omega=2*pi/T0，对 beta 作有界网格搜索。",
                    "每个 beta 下最小二乘求 A、B；以轨迹残差、包络下降和 beta 物理域共同判断可信度。",
                ),
            )
        },
    ),
    "v2_A": _experiment(
        "摆线轨道周期运动",
        "Cycloid-track periodic motion",
        ("time_s", "x_m"),
        "x(t) is fitted by three harmonics of omega; omega^2 = g/(4*a)",
        r"x(t)\approx c_0+\sum_{k=1}^{3}[a_k\cos(k\omega t)+b_k\sin(k\omega t)],\quad\omega^2=g/(4a)",
        "_fit_v2a",
        "per_complete_cycle_three_harmonic_period_search",
        "按同相位极值切出完整摆线周期，逐周期搜索三阶谐波主频并汇总重力估计。",
        {
            "gravity_g": _parameter(
                "重力加速度",
                "Gravitational acceleration",
                "g",
                "m/s^2",
                "g_hat = 4*a*omega_hat^2",
                r"\hat g=4a\hat\omega^2",
                (
                    "跳过静止前缀，用同相位极值将轨迹切成一个或多个完整摆线周期。",
                    "每个周期在有效重力限定的频率区间搜索 omega，并拟合三阶谐波。",
                    "逐周期计算 g_i=4*a*omega_i^2，取中位数并检查周期内残差与跨周期一致性。",
                ),
            )
        },
    ),
    "v2_B": _experiment(
        "双墙重复碰撞",
        "Repeated two-wall impacts",
        ("time_s", "x_m"),
        "each guarded between-impact x segment is linear; e_i = abs(v_post_i/v_pre_i)",
        r"x_i(t)\approx a_i+v_it,\quad e_i=|v_{i,+}/v_{i,-}|",
        "_fit_v2b",
        "per_wall_impact_guarded_velocity_ratios",
        "用冻结的左右墙位置定位交替碰撞，在线性墙间段上逐次计算速度比并取中位数。",
        {
            "restitution_e": _parameter(
                "恢复系数",
                "Coefficient of restitution",
                "e",
                "1",
                "e_hat = median(e_i), where e_i = abs(v_post_i/v_pre_i)",
                r"\hat e=\operatorname{median}_i|v_{i,+}/v_{i,-}|",
                (
                    "用冻结的左右墙面筛选真实反向事件，并排除每次碰撞附近的保护帧。",
                    "对各墙间运动段作线性拟合，逐碰撞取得 v_pre、v_post 与 e_i。",
                    "取有效 e_i 的中位数，并检查墙面交替、段内匀速、瞬时接触、物理域和跨碰撞一致性。",
                ),
            )
        },
    ),
    "v2_C": _experiment(
        "双表面分段摩擦",
        "Two-surface piecewise friction",
        ("time_s", "x_m"),
        "x_j(tau) = c_j0 + c_j1*tau + c_j2*tau^2; a_j = 2*c_j2",
        r"x_j(\tau)=c_{j0}+c_{j1}\tau+c_{j2}\tau^2,\quad a_j=2c_{j2},\ j\in\{A,B\}",
        "_fit_v2c",
        "two_surface_segmented_quadratic_with_velocity_continuity",
        "以带滞回的唯一表面跨越切出 A、B 两段，去除静止尾段后分别拟合减速度并检查速度连续。",
        {
            "kinetic_friction_mu_A": _parameter(
                "表面 A 动摩擦系数",
                "Surface-A kinetic-friction coefficient",
                "mu_A",
                "1",
                "mu_A_hat = -2*c_A2/(N/m)",
                r"\hat\mu_A=-2c_{A2}/(N/m)",
                (
                    "用带滞回的已知 x 分界确认恰好一次 A→B 跨越，并选择表面 A 的运动段。",
                    "对 A 段作稳健二次拟合，得到 a_A=2*c_A2。",
                    "用已知 N/m 从 a_A=-(N/m)*mu_A 反解，并检查残差、恒减速、mu_A>=0 和跨界速度连续。",
                ),
            ),
            "kinetic_friction_mu_B": _parameter(
                "表面 B 动摩擦系数",
                "Surface-B kinetic-friction coefficient",
                "mu_B",
                "1",
                "mu_B_hat = -2*c_B2/(N/m)",
                r"\hat\mu_B=-2c_{B2}/(N/m)",
                (
                    "从唯一 A→B 跨越后选择表面 B，并去除停止后的静止尾段。",
                    "对 B 段作稳健二次拟合，得到 a_B=2*c_B2。",
                    "用已知 N/m 从 a_B=-(N/m)*mu_B 反解，并检查残差、恒减速、mu_B>=0 和跨界速度连续。",
                ),
            ),
        },
    ),
    "v2_D": _experiment(
        "重力—阻尼摆",
        "Gravity-damping pendulum",
        ("time_s", "x_m", "z_m", "theta_rad"),
        "theta_ddot + 2*beta*theta_dot + (g/L)*sin(theta) = 0",
        r"\ddot\theta+2\beta\dot\theta+(g/L)\sin\theta=0",
        "_fit_v2d",
        "segmented_integral_regression_with_forward_trajectory_validation",
        "只使用一个或多个完整摆动周期做双积分稳健回归，并用前向积分轨迹复核重力与阻尼。",
        {
            "gravity_g": _parameter(
                "重力加速度",
                "Gravitational acceleration",
                "g",
                "m/s^2",
                "g_hat = coefficient of the double-integrated gravity basis",
                r"\hat g=c_g",
                (
                    "由已知枢轴和摆长把 x,z 转为展开摆角，并截取完整摆动周期。",
                    "在完整周期上构造 -(1/L) double_integral(sin(theta)) 的重力基并与阻尼基联合回归。",
                    "取重力基系数，并用该 g、beta 前向积分验证整段 theta(t) 与周期稳定性。",
                ),
            ),
            "linear_damping_beta": _parameter(
                "线性阻尼率",
                "Linear damping rate",
                "beta",
                "1/s",
                "beta_hat = coefficient of the -2*integral(theta-theta0) damping basis",
                r"\hat\beta=c_\beta",
                (
                    "由 x,z 计算展开摆角并仅保留完整摆动周期。",
                    "构造 -2*integral(theta-theta0) 的阻尼基，与重力基联合稳健回归。",
                    "取阻尼系数，并检查衰减支持、物理域以及前向积分轨迹残差。",
                ),
            ),
        },
    ),
    "v2_E": _experiment(
        "重力—恢复系数弹跳",
        "Gravity-restitution bouncing",
        ("time_s", "z_m"),
        "z_j(tau) = z_j0 + v_j0*tau - 0.5*g*tau^2",
        r"z_j(\tau)=z_{j0}+v_{j0}\tau-\tfrac12g\tau^2",
        "_fit_v2e",
        "per_flight_ballistic_curvature_and_per_impact_velocity_ratio",
        "用地面接触状态机划分逐段弹道，对各腾空段作弹道检查并逐碰撞估计恢复系数。",
        {
            "gravity_g": _parameter(
                "重力加速度",
                "Gravitational acceleration",
                "g",
                "m/s^2",
                "g_hat = shared ballistic curvature across all flight segments",
                r"\hat g=c_{\mathrm{shared}}",
                (
                    "用地面接触状态机识别着地事件并切出相邻碰撞之间的腾空段。",
                    "每段保留独立初始位置、速度，在所有有效腾空段间共享重力曲率。",
                    "联合回归得到 g_hat，并分别检查每段弹道残差、恒加速度及共享重力的物理域。",
                ),
            ),
            "restitution_e": _parameter(
                "恢复系数",
                "Coefficient of restitution",
                "e",
                "1",
                "e_hat = median(-v_post/v_pre) over valid impacts",
                r"\hat e=\operatorname{median}_i(-v_{i,\mathrm{post}}/v_{i,\mathrm{pre}})",
                (
                    "由相邻的碰撞前、后腾空段计算每次着地前后的竖直速度。",
                    "仅保留下落 v_pre<0、反弹 v_post>0 且没有过长接触停留的事件。",
                    "取有效速度比中位数，并检查 0<=e<=1 与跨碰撞一致性。",
                ),
            ),
        },
    ),
    "v3_A": _experiment(
        "带线性阻力的抛射与弹跳",
        "Linear-drag projectile with bounce",
        ("time_s", "x_m", "z_m"),
        "x(tau)=c0+c1*(1-exp(-beta*tau)); vertical flights share the same beta",
        r"x(\tau)=c_0+c_1(1-e^{-\beta\tau})",
        "_fit_v3a",
        "separate_horizontal_drag_and_vertical_gravity_restitution_components",
        "按坐标轴拆分证据：水平运动估计线性阻力，竖直腾空段估计重力，地面碰撞估计恢复系数。",
        {
            "gravity_g": _parameter(
                "重力加速度",
                "Gravitational acceleration",
                "g",
                "m/s^2",
                "g_hat = -beta_hat*s_hat, where s_hat is the shared linear-time coefficient in z",
                r"\hat g=-\hat\beta\hat s",
                (
                    "先从水平位移估计共享线性阻力率 beta_hat。",
                    "以 z=z0+A*(1-exp(-beta*tau))+s*tau 联合拟合竖直腾空段。",
                    "由线性阻力弹道关系计算 g_hat=-beta_hat*s_hat。",
                ),
            ),
            "linear_drag_beta": _parameter(
                "线性阻力率",
                "Linear-drag rate",
                "beta",
                "1/s",
                "beta_hat = argmin_beta MSE(x, c0+c1*(1-exp(-beta*tau)))",
                r"\hat\beta=\arg\min_\beta\operatorname{MSE}[x,c_0+c_1(1-e^{-\beta\tau})]",
                (
                    "在预设物理范围内枚举 beta。",
                    "每个 beta 下线性最小二乘求 c0、c1。",
                    "选择水平轨迹均方误差最小的 beta。",
                ),
            ),
            "restitution_e": _parameter(
                "恢复系数",
                "Coefficient of restitution",
                "e",
                "1",
                "e_hat = median(-v_post/v_pre) from drag-aware vertical flights",
                r"\hat e=\operatorname{median}_i(-v_{i,\mathrm{post}}/v_{i,\mathrm{pre}})",
                (
                    "检测竖直轨迹中的着地极小值。",
                    "由带阻力的分段弹道计算每次碰撞前后速度。",
                    "取有效反弹速度比的中位数。",
                ),
            ),
        },
    ),
    "v3_B": _experiment(
        "摩擦与双墙非对称碰撞",
        "Friction with asymmetric wall restitution",
        ("time_s", "x_m"),
        "x_j(tau)=x_j0+v_j0*tau-0.5*mu*g*sign(v_j)*tau^2",
        r"x_j(\tau)=x_{j0}+v_{j0}\tau-\tfrac12\mu_k g\,\operatorname{sign}(v_j)\tau^2",
        "_fit_v3b",
        "per_run_friction_and_per_wall_impact_restitution",
        "按冻结墙面碰撞切分墙间运动段，逐段估计摩擦并分别汇总左右墙恢复系数。",
        {
            "kinetic_friction_mu_k": _parameter(
                "动摩擦系数",
                "Kinetic-friction coefficient",
                "mu_k",
                "1",
                "mu_hat = shared signed-deceleration coefficient with known g",
                r"\hat\mu_k=c_\mu",
                (
                    "只保留靠近冻结左右墙的真实碰撞，并据此切分各墙间运动段。",
                    "每段独立作二次拟合，由减速度反解一个 mu_j。",
                    "对通过残差、恒减速和物理域检查的 mu_j 取中位数，并检查跨段一致性。",
                ),
            ),
            "left_restitution_e_L": _parameter(
                "左墙恢复系数",
                "Left-wall restitution",
                "e_L",
                "1",
                "e_L_hat = median(abs(v_after/v_before)) for left-wall impacts",
                r"\hat e_L=\operatorname{median}_{i\in L}|v_{i,+}/v_{i,-}|",
                (
                    "由左墙相邻两侧的独立运动段得到碰撞前后速度。",
                    "仅保留左墙位置、接触停留和 0<=e<=1 都通过的事件。",
                    "取左墙有效速度幅值比的中位数，并检查跨事件一致性。",
                ),
            ),
            "right_restitution_e_R": _parameter(
                "右墙恢复系数",
                "Right-wall restitution",
                "e_R",
                "1",
                "e_R_hat = median(abs(v_after/v_before)) for right-wall impacts",
                r"\hat e_R=\operatorname{median}_{i\in R}|v_{i,+}/v_{i,-}|",
                (
                    "由右墙相邻两侧的独立运动段得到碰撞前后速度。",
                    "仅保留右墙位置、接触停留和 0<=e<=1 都通过的事件。",
                    "取右墙有效速度幅值比的中位数，并检查跨事件一致性。",
                ),
            ),
        },
    ),
    "v3_C": _experiment(
        "斜坡—地面—墙面组合运动",
        "Ramp-floor-wall hybrid dynamics",
        ("time_s", "x_m", "z_m", "ramp_s_m"),
        "a_ramp=g*sin(alpha)-mu*g*cos(alpha); a_floor=mu*g; e=abs(v_post/v_pre)",
        r"a_r=g\sin\alpha-\mu g\cos\alpha,\quad a_f=\mu g,\quad e=|v_+/v_-|",
        "_fit_v3c",
        "separate_ramp_prewall_floor_wall_impact_and_postwall_floor",
        "按装置几何依次切出斜坡、撞墙前地面、墙面碰撞和撞墙后地面，分别构造三个参数证据。",
        {
            "gravity_g": _parameter(
                "重力加速度",
                "Gravitational acceleration",
                "g",
                "m/s^2",
                "g_hat = (a_ramp + a_floor*cos(alpha))/sin(alpha)",
                r"\hat g=(a_r+a_f\cos\alpha)/\sin\alpha",
                (
                    "沿已知斜坡方向投影轨迹并二次拟合得到 a_ramp。",
                    "分别拟合撞墙前、后的地面减速度，并以两者中位数得到共享 a_floor。",
                    "由 a_ramp=g*sin(alpha)-mu*g*cos(alpha) 与 a_floor=mu*g 联立反解 g_hat。",
                ),
            ),
            "kinetic_friction_mu": _parameter(
                "动摩擦系数",
                "Kinetic-friction coefficient",
                "mu_k",
                "1",
                "mu_hat = a_floor/g_hat",
                r"\hat\mu_k=a_f/\hat g",
                (
                    "分别从撞墙前、后的地面段二次拟合减速度并检查两段一致性。",
                    "使用共享 a_floor 与斜坡段联合反解 g_hat。",
                    "由 mu_hat=a_floor/g_hat 得到摩擦系数。",
                ),
            ),
            "restitution_e": _parameter(
                "恢复系数",
                "Coefficient of restitution",
                "e",
                "1",
                "e_hat = abs(v_post/v_pre) at the known wall",
                r"\hat e=|v_{\mathrm{post}}/v_{\mathrm{pre}}|",
                (
                    "在斜坡→地面之后、已知右墙附近定位一次水平速度反向事件。",
                    "排除接触邻域后用局部轨迹导数估计 v_pre 与 v_post，并检查接触停留。",
                    "取 e_hat=|v_post/v_pre| 并检查 0<=e<=1。",
                ),
            ),
        },
    ),
    "v3_D": _experiment(
        "重力—阻尼—磁力摆",
        "Gravity-damping-magnetic pendulum",
        ("time_s", "x_m", "z_m", "theta_rad"),
        "theta_ddot=-(g/L)*sin(theta)-2*beta*theta_dot+kappa*h(theta); h=-u*exp(-u^2/2)",
        r"\ddot\theta=-(g/L)\sin\theta-2\beta\dot\theta+\kappa h(\theta),\quad h=-u e^{-u^2/2}",
        "_fit_v3d",
        "complete_cycles_three_basis_regression_with_forward_identifiability_gate",
        "只在完整磁摆周期上做重力、阻尼、磁力三基稳健回归，并以前向轨迹和可辨识性门复核。",
        {
            "gravity_g": _parameter(
                "重力加速度",
                "Gravitational acceleration",
                "g",
                "m/s^2",
                "g_hat = coefficient of the double-integrated gravity basis",
                r"\hat g=c_g",
                (
                    "由已知枢轴和摆长计算展开摆角，并只保留一个或多个完整磁摆周期。",
                    "在完整周期上构造重力、阻尼和磁力三个双积分基。",
                    "联合稳健回归取重力系数，并以前向轨迹残差、周期稳定性和设计矩阵可辨识性复核。",
                ),
            ),
            "linear_damping_beta": _parameter(
                "线性阻尼率",
                "Linear damping rate",
                "beta",
                "1/s",
                "beta_hat = coefficient of the double-integrated damping basis",
                r"\hat\beta=c_\beta",
                (
                    "由轨迹计算展开摆角并只保留完整磁摆周期。",
                    "构造 -2*integral(theta-theta0) 的阻尼基。",
                    "与重力、磁力基联合回归取阻尼系数，并检查物理域、前向轨迹和可辨识性。",
                ),
            ),
            "magnetic_kappa": _parameter(
                "磁力强度系数",
                "Magnetic-strength coefficient",
                "kappa",
                "rad/s^2",
                "kappa_hat = coefficient of the double-integrated known magnetic profile h(theta)",
                r"\hat\kappa=c_\kappa",
                (
                    "在完整周期内按已知磁体中心和宽度计算 u 与局部磁力形状 h(theta)。",
                    "对 h(theta) 构造双积分磁力基。",
                    "与重力、阻尼基联合回归取磁力系数，并要求实际穿越磁区且三基可辨识。",
                ),
            ),
        },
    ),
}


def get_formula_spec(experiment_id: str, parameter_name: str) -> dict[str, Any]:
    """Return one self-contained formula specification.

    A deep copy is returned so report-specific labels can be added without
    mutating the process-wide registry.  Unknown experiment and parameter IDs
    raise ``KeyError`` with a precise message.
    """

    experiment_key = str(experiment_id)
    parameter_key = str(parameter_name)
    if experiment_key not in FORMULA_REGISTRY:
        raise KeyError(f"unknown experiment formula: {experiment_key!r}")
    experiment = FORMULA_REGISTRY[experiment_key]
    parameters = experiment["parameters"]
    if parameter_key not in parameters:
        raise KeyError(f"unknown formula axis: {experiment_key}/{parameter_key}")
    shared = {key: value for key, value in experiment.items() if key != "parameters"}
    return copy.deepcopy(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_key,
            "parameter_name": parameter_key,
            **shared,
            **parameters[parameter_key],
        }
    )


def experiment_system_count() -> int:
    """Return the number of distinct frozen physical experiment systems."""

    return len(EXPERIMENT_ORDER)


def parameter_channel_manifest() -> list[dict[str, Any]]:
    """Return the ordered experiment-parameter response-channel manifest.

    The unit represented by one row is an OAT response channel, not a new
    experiment.  For example, V2_D contributes one experiment system and two
    channels (``gravity_g`` and ``linear_damping_beta``).
    """

    channels: list[dict[str, Any]] = []
    for experiment_index, experiment_id in enumerate(EXPERIMENT_ORDER, start=1):
        experiment = FORMULA_REGISTRY[experiment_id]
        for parameter_name, parameter in experiment["parameters"].items():
            channels.append(
                {
                    "channel_index": len(channels) + 1,
                    "channel_id": f"{experiment_id}/{parameter_name}",
                    "experiment_index": experiment_index,
                    "experiment_id": experiment_id,
                    "experiment_title_zh": experiment["experiment_title_zh"],
                    "experiment_title_en": experiment["experiment_title_en"],
                    "parameter_name": parameter_name,
                    "parameter_title_zh": parameter["parameter_title_zh"],
                    "parameter_title_en": parameter["parameter_title_en"],
                    "symbol": parameter["symbol"],
                    "unit": parameter["unit"],
                }
            )
    return copy.deepcopy(channels)


def benchmark_scope_summary() -> dict[str, Any]:
    """Return explicit counting units for report headers and metadata."""

    channels = parameter_channel_manifest()
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_system_count": experiment_system_count(),
        "parameter_response_channel_count": len(channels),
        "counting_rule_zh": (
            "13 表示不同的物理实验系统；24 表示“实验系统 × 单个待扫描参数”的 OAT 参数响应通道。"
            "24 不是实验数量；多参数系统贡献多个通道，但不会因此被重复计为多个实验。"
        ),
        "counting_rule_en": (
            "13 counts distinct physical experiment systems; 24 counts one-at-a-time "
            "experiment-parameter response channels. A multi-parameter system contributes "
            "multiple channels but remains one experiment."
        ),
        "parameter_channels_per_experiment": {
            experiment_id: len(FORMULA_REGISTRY[experiment_id]["parameters"])
            for experiment_id in EXPERIMENT_ORDER
        },
    }


def scan_axis_count() -> int:
    """Return the frozen number of OAT parameter-response channels."""

    return len(parameter_channel_manifest())


def _finite_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return int(value) if isinstance(value, int) else number


def _quality_records(value: Any, path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        fit_nrmse = _finite_number(value.get("fit_nrmse"))
        fit_points = _finite_number(value.get("fit_points"))
        if fit_nrmse is not None or fit_points is not None:
            records.append(
                {
                    "path": path,
                    "fit_nrmse": fit_nrmse,
                    "fit_points": fit_points,
                }
            )
        for key, item in value.items():
            # fit_series contains long observed/predicted vectors, never a
            # second fit-quality record.  Avoid traversing it in report code.
            if str(key) == "fit_series":
                continue
            records.extend(_quality_records(item, f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            records.extend(_quality_records(item, f"{path}[{index}]"))
    return records


def extract_fit_quality_records(diagnostics: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Recursively extract compact ``fit_nrmse``/``fit_points`` records."""

    if not isinstance(diagnostics, Mapping):
        return []
    return _quality_records(diagnostics, "diagnostics")


READABLE_INTERMEDIATE_KEYS: tuple[str, ...] = (
    "fit_rmse",
    "fit_r2",
    "fit_parameter_count",
    "estimated_acceleration_m_s2",
    "moving_fit_points",
    "estimated_period_s",
    "turning_point_count",
    "impact_count",
    "impact_time_s",
    "v_pre",
    "v_post",
    "velocity_ratio",
    "acceleration_A_m_s2",
    "acceleration_B_m_s2",
    "ramp_acceleration_m_s2",
    "floor_deceleration_m_s2",
    "design_matrix_rank",
    "normalized_condition_number",
    "magnet_zone_transition_count",
)


def _intermediate_records(value: Any, path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            number = _finite_number(item)
            if key_text in READABLE_INTERMEDIATE_KEYS and number is not None:
                records.append({"path": child_path, "name": key_text, "value": number})
            if key_text != "fit_series":
                records.extend(_intermediate_records(item, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            records.extend(_intermediate_records(item, f"{path}[{index}]"))
    return records


def extract_readable_intermediates(
    diagnostics: Mapping[str, Any] | None,
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Extract a bounded list of human-readable scalar fit intermediates."""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    if not isinstance(diagnostics, Mapping) or limit == 0:
        return []
    return _intermediate_records(diagnostics, "diagnostics")[:limit]


def summarize_fit_diagnostics(
    diagnostics: Mapping[str, Any] | None,
    *,
    intermediate_limit: int = 12,
) -> dict[str, Any]:
    """Return compact quality and intermediate evidence for a report card."""

    quality = extract_fit_quality_records(diagnostics)
    nrmse_values = [float(row["fit_nrmse"]) for row in quality if row["fit_nrmse"] is not None]
    point_values = [int(row["fit_points"]) for row in quality if row["fit_points"] is not None]
    return {
        "quality_records": quality,
        "quality_record_count": len(quality),
        "worst_fit_nrmse": max(nrmse_values) if nrmse_values else None,
        "minimum_fit_points": min(point_values) if point_values else None,
        "intermediates": extract_readable_intermediates(diagnostics, limit=intermediate_limit),
    }


__all__ = [
    "EXPERIMENT_ORDER",
    "FORMULA_REGISTRY",
    "READABLE_INTERMEDIATE_KEYS",
    "SCHEMA_VERSION",
    "benchmark_scope_summary",
    "experiment_system_count",
    "extract_fit_quality_records",
    "extract_readable_intermediates",
    "get_formula_spec",
    "parameter_channel_manifest",
    "scan_axis_count",
    "summarize_fit_diagnostics",
]
