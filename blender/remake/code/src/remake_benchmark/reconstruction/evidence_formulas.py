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


SCHEMA_VERSION = "1.0.0"


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
        "robust_airborne_quadratic",
        "只取第一次连续腾空段，对高度作稳健二次回归。",
        {
            "gravity_g": _parameter(
                "重力加速度",
                "Gravitational acceleration",
                "g",
                "m/s^2",
                "g_hat = -2*c2",
                r"\hat g=-2c_2",
                (
                    "按接触高度截取第一次连续腾空段。",
                    "以首个有效时刻为零点，将 z 对 tau 作稳健二次拟合。",
                    "由二次项系数 c2 计算 g_hat=-2*c2。",
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
        "robust_piecewise_linear_velocity_ratio",
        "搜索一次速度反向点，并分别回归碰撞前后的水平速度。",
        {
            "restitution_e": _parameter(
                "恢复系数",
                "Coefficient of restitution",
                "e",
                "1",
                "e_hat = abs(v_post/v_pre)",
                r"\hat e=\left|v_{\mathrm{post}}/v_{\mathrm{pre}}\right|",
                (
                    "搜索靠近已知墙面的最佳分段点。",
                    "分别拟合碰撞前后的直线斜率 v_pre 与 v_post。",
                    "用反向速度幅值比得到 e_hat。",
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
        "robust_pre_stop_quadratic",
        "在停止前的运动段上作稳健二次回归，并使用已知重力。",
        {
            "kinetic_friction_mu": _parameter(
                "动摩擦系数",
                "Kinetic-friction coefficient",
                "mu_k",
                "1",
                "mu_hat = -a/g_known = -2*c2/g_known",
                r"\hat\mu_k=-a/g_{\mathrm{known}}=-2c_2/g_{\mathrm{known}}",
                (
                    "根据正向速度截取停止前运动段。",
                    "对 x(tau) 作稳健二次拟合并取 a=2*c2。",
                    "由 a=-mu*g_known 反解动摩擦系数。",
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
        "fixed_period_bounded_decay_search",
        "由已知枢轴换算摆角，在固定周期下搜索最小残差的衰减率。",
        {
            "amplitude_decay_beta": _parameter(
                "振幅衰减率",
                "Amplitude-decay rate",
                "beta",
                "1/s",
                "beta_hat = argmin_beta MSE(theta, theta_model(beta))",
                r"\hat\beta=\arg\min_{\beta}\operatorname{MSE}(\theta,\theta_{\mathrm{model}}(\beta))",
                (
                    "用球心与已知枢轴计算并展开角度 theta。",
                    "固定 omega=2*pi/T0，对 beta 作有界网格搜索。",
                    "每个 beta 下最小二乘求 A、B，选择残差最小者。",
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
        "three_harmonic_bounded_period_search",
        "用三阶谐波模型搜索主频，再利用已知摆线尺度反解重力。",
        {
            "gravity_g": _parameter(
                "重力加速度",
                "Gravitational acceleration",
                "g",
                "m/s^2",
                "g_hat = 4*a*omega_hat^2",
                r"\hat g=4a\hat\omega^2",
                (
                    "在由有效重力范围限定的频率区间内搜索 omega。",
                    "每个频率下拟合三阶谐波并选择轨迹残差最小者。",
                    "使用已知摆线尺度 a 计算 g_hat=4*a*omega_hat^2。",
                ),
            )
        },
    ),
    "v2_B": _experiment(
        "双墙重复碰撞",
        "Repeated two-wall impacts",
        ("time_s", "x_m"),
        "each between-impact x segment is linear; e_i = abs(v_(i+1)/v_i)",
        r"x_i(t)\approx a_i+v_it,\quad e_i=|v_{i+1}/v_i|",
        "_fit_v2b",
        "piecewise_line_geometric_mean_ratio",
        "按转向点切分直线运动段，用多次碰撞速度比的几何平均估计恢复系数。",
        {
            "restitution_e": _parameter(
                "恢复系数",
                "Coefficient of restitution",
                "e",
                "1",
                "e_hat = exp(mean(log(e_i)))",
                r"\hat e=\exp\left(\frac{1}{N}\sum_i\log e_i\right)",
                (
                    "从 x(t) 的转向点切分墙间运动段。",
                    "对每段作线性拟合得到速度 v_i。",
                    "取相邻反向速度比 e_i 的几何平均。",
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
        "two_segment_robust_quadratic",
        "按已知表面分界把轨迹切成 A、B 两段，并分别拟合减速度。",
        {
            "kinetic_friction_mu_A": _parameter(
                "表面 A 动摩擦系数",
                "Surface-A kinetic-friction coefficient",
                "mu_A",
                "1",
                "mu_A_hat = -2*c_A2/(N/m)",
                r"\hat\mu_A=-2c_{A2}/(N/m)",
                (
                    "用已知 x 分界选择表面 A 的轨迹点。",
                    "对 A 段作稳健二次拟合，得到 a_A=2*c_A2。",
                    "用已知 N/m 从 a_A=-(N/m)*mu_A 反解。",
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
                    "用已知 x 分界选择表面 B，并去除停止后的静止尾段。",
                    "对 B 段作稳健二次拟合，得到 a_B=2*c_B2。",
                    "用已知 N/m 从 a_B=-(N/m)*mu_B 反解。",
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
        "double_integral_robust_regression",
        "将摆方程双积分后，对重力基与阻尼基进行稳健线性回归。",
        {
            "gravity_g": _parameter(
                "重力加速度",
                "Gravitational acceleration",
                "g",
                "m/s^2",
                "g_hat = coefficient of the double-integrated gravity basis",
                r"\hat g=c_g",
                (
                    "由已知枢轴和摆长把 x,z 转为展开摆角 theta。",
                    "构造 -(1/L) double_integral(sin(theta)) 的重力基。",
                    "与阻尼基联合稳健回归，取重力基系数。",
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
                    "由 x,z 计算展开摆角 theta。",
                    "构造 -2*integral(theta-theta0) 的阻尼基。",
                    "与重力基联合稳健回归，取阻尼基系数。",
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
        "shared_ballistic_curvature_and_impact_velocity_ratio",
        "检测着地点，在多个腾空段间共享重力曲率，再由碰撞速度比估计恢复系数。",
        {
            "gravity_g": _parameter(
                "重力加速度",
                "Gravitational acceleration",
                "g",
                "m/s^2",
                "g_hat = shared ballistic curvature across all flight segments",
                r"\hat g=c_{\mathrm{shared}}",
                (
                    "用高度局部极小值识别着地事件并划分腾空段。",
                    "为每段保留独立初始位置、速度，同时共享重力曲率。",
                    "对全部腾空点联合稳健回归得到 g_hat。",
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
                    "由联合弹道模型计算每次着地前后的竖直速度。",
                    "仅保留下落 v_pre<0 且反弹 v_post>0 的事件。",
                    "取所有有效速度比的中位数。",
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
        "horizontal_exponential_plus_shared_drag_flights",
        "先由水平指数衰减搜索阻力率，再将该阻力率用于竖直分段弹道和碰撞估计。",
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
        "shared_segment_deceleration_and_wall_specific_ratios",
        "对全部墙间运动段共享摩擦减速度，并按左右墙分别汇总碰撞速度比。",
        {
            "kinetic_friction_mu_k": _parameter(
                "动摩擦系数",
                "Kinetic-friction coefficient",
                "mu_k",
                "1",
                "mu_hat = shared signed-deceleration coefficient with known g",
                r"\hat\mu_k=c_\mu",
                (
                    "按 x 转向点划分左右运动段。",
                    "为各段设置独立位置和初速度，并共享 mu*g 的减速度项。",
                    "对所有运动段联合稳健回归得到 mu_hat。",
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
                    "从分段模型得到各次碰撞前后的速度。",
                    "按碰撞位置将事件归入左墙。",
                    "取左墙事件速度幅值比的中位数。",
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
                    "从分段模型得到各次碰撞前后的速度。",
                    "按碰撞位置将事件归入右墙。",
                    "取右墙事件速度幅值比的中位数。",
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
        "ramp_floor_acceleration_decomposition_and_wall_ratio",
        "分别拟合斜坡和水平地面加速度，再用墙面碰撞速度比估计恢复系数。",
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
                    "对水平地面正向运动段二次拟合得到减速度 a_floor。",
                    "联立两段动力学方程反解 g_hat。",
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
                    "从水平地面段二次拟合得到减速度 a_floor。",
                    "由斜坡和地面加速度联合反解 g_hat。",
                    "使用 a_floor=mu*g 计算 mu_hat。",
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
                    "在已知墙面附近定位水平速度反向事件。",
                    "分别用碰撞前后局部多项式导数估计速度。",
                    "取反向速度幅值比得到 e_hat。",
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
        "three_basis_double_integral_robust_regression",
        "将重力、线性阻尼和已知形状的磁力项双积分后进行三基稳健回归。",
        {
            "gravity_g": _parameter(
                "重力加速度",
                "Gravitational acceleration",
                "g",
                "m/s^2",
                "g_hat = coefficient of the double-integrated gravity basis",
                r"\hat g=c_g",
                (
                    "由已知枢轴和摆长计算展开摆角 theta。",
                    "构造重力、阻尼和磁力三个双积分基。",
                    "联合稳健回归并取重力基系数。",
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
                    "由轨迹计算展开摆角 theta。",
                    "构造 -2*integral(theta-theta0) 的阻尼基。",
                    "与重力、磁力基联合回归并取阻尼系数。",
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
                    "按已知磁体中心和宽度计算 u 与磁力形状 h(theta)。",
                    "对 h(theta) 构造双积分磁力基。",
                    "与重力、阻尼基联合回归并取磁力基系数。",
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


def scan_axis_count() -> int:
    """Return the frozen number of experiment-parameter response axes."""

    return sum(len(experiment["parameters"]) for experiment in FORMULA_REGISTRY.values())


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
    "FORMULA_REGISTRY",
    "READABLE_INTERMEDIATE_KEYS",
    "SCHEMA_VERSION",
    "extract_fit_quality_records",
    "extract_readable_intermediates",
    "get_formula_spec",
    "scan_axis_count",
    "summarize_fit_diagnostics",
]
