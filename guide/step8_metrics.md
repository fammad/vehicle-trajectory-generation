# Step 8 · Metrics

**Where:** Colab, same notebook · **Time:** about 1.5 hours

**Goal:** turn "it looks better" into numbers. Two questions:
- **Feasibility:** are the rules kept?
- **Fidelity:** do the paths still look like real driving?

A strategy that keeps every rule but makes all cars drive the same boring path is not good. We need both numbers.

Markdown cell: `## 6 · Metrics` and this table:

| Metric | Question | Good value |
|---|---|---|
| Violation (per rule) | how much is a rule broken on average? (m, m/s, m/s²) | 0 |
| On-surface rate | share of paths that break no rule | high (real data: 99%) |
| Excess SWD | how different is the cloud of generated paths from the cloud of real paths? | 0 |
| Road-edge W1 | do generated points sit at the same distance from the road edge as real ones? | 0 |
| Diversity | for one history, how spread out are 20 generated end points? (m) | close to real behaviour, not 0 |
| Jerk p95 | how sharp are the kinks? (m/s³) | close to real data |
| Manoeuvre shares | left / straight / right / stopped | close to real data |

---

### Cell 1 · Helper functions

```python
from scipy.stats import wasserstein_distance

def sliced_wasserstein(a, b, n_directions=200):
    # a, b: (N, 60). Look at both clouds from 200 random directions and compare the sorted values.
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(a.shape[1], n_directions))
    directions /= np.linalg.norm(directions, axis=0)
    a_sorted = np.sort(a @ directions, axis=0)
    b_sorted = np.sort(b @ directions, axis=0)
    return np.abs(a_sorted - b_sorted).mean()

def to_vehicle_np(p_map, origin, yaw):
    # numpy version: map points (N, 30, 2) -> vehicle frame (N, 30, 2)
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    dx = p_map[..., 0] - origin[:, None, 0]
    dy = p_map[..., 1] - origin[:, None, 1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=-1)

def manoeuvre_shares(fut_veh):
    end = fut_veh[:, -1]                                        # end point seen from the car
    angle = np.degrees(np.arctan2(end[:, 1], end[:, 0]))
    label = np.where(angle > 30, "left", np.where(angle < -30, "right", "straight"))
    label = np.where(np.linalg.norm(end, axis=1) < 2, "stopped", label)
    return {name: np.mean(label == name) for name in ["left", "straight", "right", "stopped"]}
```

- **Wasserstein distance (W1)** between two lists of numbers: sort both and take the average difference. "How much do I have to move one pile of sand to get the other pile."
- **Sliced Wasserstein (SWD)** does this for 60-number paths: project both clouds onto many random directions (each projection is just a list of numbers), compare, average.
- `@` is matrix multiplication.
- Manoeuvre: the angle of the end point. More than 30° left = left turn, and so on.

---

### Cell 2 · One function that measures everything

```python
def evaluate(paths, real, p0, origin, yaw):
    # paths, real: (N, 30, 2) torch tensors in map coordinates, for the same N histories
    g_road, g_speed, g_acc = rule_values(paths, p0)
    gen = paths.cpu().numpy()
    real_np = real.cpu().numpy()
    o, y = origin.cpu().numpy(), yaw.cpu().numpy()
    gen_veh = to_vehicle_np(gen, o, y).reshape(len(gen), -1)
    real_veh = to_vehicle_np(real_np, o, y).reshape(len(gen), -1)

    # excess SWD: generated vs real, minus real vs real (two halves), same size
    half = len(gen) // 2
    swd_gen = sliced_wasserstein(gen_veh[:half], real_veh[half:2 * half])
    swd_real = sliced_wasserstein(real_veh[:half], real_veh[half:2 * half])

    jerk = np.linalg.norm(np.diff(gen, n=3, axis=1), axis=-1) / DT**3
    result = {
        "violation road (m)": torch.relu(g_road).mean().item(),
        "violation speed (m/s)": torch.relu(g_speed).mean().item(),
        "violation acc (m/s²)": torch.relu(g_acc).mean().item(),
        "on-surface rate": is_feasible(paths, p0).float().mean().item(),
        "excess SWD": swd_gen - swd_real,
        "road-edge W1 (m)": wasserstein_distance(road_distance(paths).cpu().numpy().ravel(),
                                                 road_distance(real).cpu().numpy().ravel()),
        "jerk p95 (m/s³)": np.percentile(jerk, 95),
    }
    for name, share in manoeuvre_shares(gen_veh.reshape(-1, F, 2)).items():
        result["share " + name] = share
    return result

def diversity(model, strategy, lam, n_histories=100, n_samples=20):
    # average distance (m) of 20 end points from their mean, for the same history
    idx = np.repeat(np.arange(n_histories), n_samples)        # 0,0,...,0,1,1,...,1,... (20 times each)
    c = c_te[idx]
    origin = torch.tensor(test["origin"][idx], device=device)
    yaw = torch.tensor(test["yaw"][idx], device=device)
    p0 = torch.tensor(test["hist_map"][idx, -1], device=device)
    ends = sample(model, c, origin, yaw, p0, strategy=strategy, lam=lam)[:, -1].cpu().numpy()
    ends = ends.reshape(n_histories, n_samples, 2)
    spread = np.linalg.norm(ends - ends.mean(axis=1, keepdims=True), axis=-1)
    return spread.mean()
```

- **Excess SWD:** even two halves of *real* data are not identical. We subtract that, so 0 means "as close to real data as real data is to itself". Small negative values are just noise.
- **Jerk:** the third difference of positions (change of change of change). High jerk = kinks. `np.percentile(jerk, 95)`: 95% of values are below this.
- `diversity`: `np.repeat` copies every history 20 times, so we can sample all 2,000 paths in one call.

---

### Cell 3 · Reference row: real data

```python
N_EVAL = 2000
origin = torch.tensor(test["origin"][:N_EVAL], device=device)
yaw = torch.tensor(test["yaw"][:N_EVAL], device=device)
p0 = torch.tensor(test["hist_map"][:N_EVAL, -1], device=device)
real = torch.tensor(test["fut_map"][:N_EVAL], device=device)
c = c_te[:N_EVAL]

reference = evaluate(real, real, p0, origin, yaw)
pd.Series(reference).round(3)
```

Real data measured with the same metrics. This is the target row in the results table. (Its excess SWD and road-edge W1 are 0 by definition.)

**For the report (Evaluation):** the metric table with one line per metric. The reference row.
