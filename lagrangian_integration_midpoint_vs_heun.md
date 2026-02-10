# 拉格朗日积分方案详细说明：Heun 与 Implicit Mid-point

本文面向当前项目中的轨迹追踪实现（`lagranto_track.py`），系统介绍两种时间推进方案：

- `Heun`（显式二阶 RK2 / 显式梯形法）
- `Implicit Mid-point`（隐式中点法，采用 Picard 迭代求解）

重点说明 `mid-point` 相对 `Heun` 的优势，以及在本项目中的具体落地意义。

---

## 1. 问题背景：我们在积分什么？

拉格朗日追踪本质是积分如下常微分方程（状态向量记为 `X = [lon, lat, z]`）：

\[
\frac{d\lambda}{dt} = \frac{u}{(R+z)\cos\phi},\quad
\frac{d\phi}{dt} = \frac{v}{R+z},\quad
\frac{dz}{dt} = w
\]

- `u, v, w` 分别是 zonal / meridional / vertical wind。
- `R` 是行星半径，`z` 是海拔高度（m）。
- 在代码里对应 `get_next_position_alt(...)` 的更新关系。
- 若 `w_positive_up=False`，则按模型约定使用 `dz/dt = -w`。

由于风场来自离散网格（`time, altitude, latitude, longitude`），每一步都需要三维插值（本项目使用 `RegularGridInterpolator`）。

---

## 2. Heun 方案

### 2.1 数值格式

对方程 `dX/dt = F(t, X)`，Heun 的一步写成：

\[
k_1 = F(t_n, X_n)
\]
\[
\tilde{X}_{n+1} = X_n + \Delta t\,k_1
\]
\[
k_2 = F(t_{n+1}, \tilde{X}_{n+1})
\]
\[
X_{n+1} = X_n + \frac{\Delta t}{2}(k_1 + k_2)
\]

### 2.2 在本项目中的实现要点

- 函数：`track_particles_heun(...)`、`track_particles_heun_backward(...)`
- 每步 2 次风场采样（`k1`, `k2`）。
- `k2` 在“预测位置 + 下一时刻风场切片”上计算。
- 在球坐标下，代码对完整状态导数 `F=[d\lambda/dt, d\phi/dt, dz/dt]` 做平均：
  - 先由 `k1_wind` 和当前状态计算 `k1 = F(t_n, X_n)`
  - 用 `k1` 预测到 `\tilde{X}_{n+1}`
  - 在 `\tilde{X}_{n+1}` 与下一时刻切片上计算 `k2 = F(t_{n+1}, \tilde{X}_{n+1})`
  - 最终按 `X_{n+1}=X_n+\Delta t (k1+k2)/2` 更新
- 不能直接平均风速再推进；因为 `d\lambda/dt = u / ((R+z)\cos\phi)` 含状态相关几何因子，
  `0.5*(u_1+u_2)` 一般不等于 `0.5*(d\lambda/dt|_1 + d\lambda/dt|_2)`。
- 高度越界或插值为 `NaN` 时，该粒子停止追踪。

### 2.3 Heun 的特点

- 优点：实现简单、计算成本低、二阶精度。
- 局限：显式法稳定域有限；长时积分下更容易出现相位漂移、数值耗散/增益误差；前后向不完全对称。

---

## 3. Implicit Mid-point 方案（当前“新方案”）

### 3.1 数值格式

隐式中点法一步写成：

\[
X_{n+1} = X_n + \Delta t\;F\left(t_n+\frac{\Delta t}{2}, \frac{X_n + X_{n+1}}{2}\right)
\]

这里 `X_{n+1}` 同时出现在等式两侧，是隐式方程，需要迭代求解。

### 3.2 在本项目中的实现流程

对应函数：`step_implicit_midpoint_alt(...)`，由 `track_particles_midpoint(...)` / `track_particles_midpoint_backward(...)` 调用。

单步流程（前向）：

1. 先构造时间中点风场切片：
   - `u_mid = 0.5*(u_now + u_next)`（`v/w` 同理）
2. 用中点切片在当前状态估计一个初值（initial guess）。
3. 进行 Picard 迭代（`picard_iters` 次）：
   - 用当前候选终点与起点求“空间中点”
   - 在该中点插值风场
   - 重新更新候选终点
4. 迭代完成后输出终点。

关键几何细节：

- 经度/纬度中点可选 `SLERP` 方式（`use_slerp_midpoint=True`）：
  - 函数 `midpoint_on_sphere(...)`
  - 避免简单经纬平均在跨日期变更线、接近极区时的几何偏差。

### 3.3 数值属性（理论层面）

隐式中点法是经典二阶方法，具有：

- 时间中心（time-centered）
- 对称性（self-adjoint, time-reversible，精确求解时）
- 对线性系统的更强稳定性（A-stable）
- 在哈密顿系统中的辛性质（symplectic）

在离散风场插值 + 有限次迭代下，这些性质会被部分削弱，但通常仍能带来更好的长时行为。

---

## 4. 重点：Mid-point 相对 Heun 的优势

以下对比是本项目最关键的工程结论。

### 4.1 长时间积分更稳，累计漂移更小

- `Heun` 虽然二阶，但属于显式预测-校正，长期多步后更容易累积相位误差与能量漂移。
- `Mid-point` 是时间中心隐式结构，对振荡/旋转型流场通常更稳，长时轨迹更“守形”。

适用场景：

- 轨迹步数很长（例如回溯/前推很多时间层）
- 关注环流结构、封闭轨道、驻波附近的保持性

### 4.2 前向/后向一致性更好（时间对称优势）

- 本项目既有前向也有后向追踪。
- `Mid-point` 本身更偏向时间对称格式，前后向结果的“互逆一致性”通常优于 `Heun`。

这对以下任务很重要：

- 溯源分析（backward tracing）
- 需要比较 forward 与 backward 结果一致性的诊断

### 4.3 对较大时间步长更鲁棒

- `Heun` 的稳定性受限于显式稳定域，步长大时更易出现误差放大或不稳定。
- `Mid-point` 通过隐式求解中点风场，通常允许在同等误差目标下使用更“激进”的 `dt`（仍需测试上限）。

工程意义：

- 在同等精度目标下，可能减少总步数（注意单步成本更高，需综合权衡）

### 4.4 球面几何一致性更好（本实现的额外优势）

`Mid-point` 实现中可启用球面中点（`SLERP`）：

- 中点在单位球上计算，再映回经纬度。
- 相比简单经纬平均，更符合球面最短路径几何。

这对两类区域收益明显：

- 跨越经度拼接处（如 `179°E -> -179°W`）
- 高纬区域（`cos(lat)` 很小，几何敏感）

### 4.5 中点采样与标量耦合更自然

项目还提供 `sample_scalar_on_midpoint_steps(...)`。  
为了避免前向/后向混用时的时间错位，采样函数采用显式时间对齐：

- `trace_time`：每一步终点状态对应的采样时间（长度必须等于 `len(mid_steps)`）
- `time_grid`：`scalar_time_interp` 第一维对应的完整时间轴（必须严格递增）

调用建议：

- 前向：
  - `mid_steps = track_particles_midpoint(...)`
  - `trace_time = get_trace_time_midpoint(time)`（通常是 `time[1:]`）
  - `sample_scalar_on_midpoint_steps(..., trace_time=trace_time, time_grid=time, ...)`

- 后向：
  - `mid_steps = track_particles_midpoint_backward(time, ..., start_index/start_time, n_steps)`
  - `trace_time = get_trace_time_midpoint_backward(time, start_index/start_time, n_steps)`
  - `sample_scalar_on_midpoint_steps(..., trace_time=trace_time, time_grid=time, ...)`

注意：`mid_steps` 与 `trace_time` 必须来自同一次追踪配置（同一组 `start_index/start_time/n_steps`）。

---

## 5. 代价与注意事项（客观对比）

Mid-point 不是“无代价升级”，主要代价如下：

- 单步成本更高：  
  - `Heun` 约 2 次风场采样/步  
  - `Mid-point` 约 `1 + picard_iters` 次采样/步（默认 `picard_iters=3` 时约 4 次）
- 迭代收敛受步长和场平滑性影响：
  - `dt` 过大、梯度过陡时，Picard 可能慢收敛或失败。
- 参数需要调优：
  - `picard_iters`
  - `picard_tol`
  - `use_slerp_midpoint`

因此更合理的结论是：  
`Mid-point` 以更高单步成本换取更好的长期稳定性与几何一致性。

---

## 6. 参数建议（本项目实践）

建议从以下配置起步，再按误差与速度做 A/B 测试：

- `picard_iters=2~4`
- `use_slerp_midpoint=True`（建议保持开启）
- `periodic_lon='auto'`（全局网格通常合理）
- 先固定 `dt` 比较 Heun 与 Mid-point 的轨迹差异，再决定是否放大 `dt`

推荐评估指标：

- 前后向闭合误差（forward-backward mismatch）
- 长时间漂移（相对参考轨迹）
- 高纬/跨 180° 轨迹连续性
- 计算耗时（每条轨迹、每千步）

---

## 7. 简短结论

如果你的目标是“短时、快速、成本敏感”，`Heun` 仍然实用。  
如果你的目标是“长时追踪、前后向一致性、球面几何可靠性”，`Implicit Mid-point` 是更优选，尤其在本项目已实现 `Picard + SLERP` 后，优势会更明显。
