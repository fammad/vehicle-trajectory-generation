# Step 3 · Road and rules

**Where:** laptop · **Notebook:** `offline/03_road_and_rules.ipynb` · **Time:** about 1.5 hours

**Goal:** write the 3 rules so a computer can check them, and check that real driving passes them.
The professor's requirement: *a constraint must be computationally checkable and say clearly if a sample is valid or not.*

---

### Cell 1 · Imports and loading

Markdown title: `# Step 3: Road and rules`

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

DATA = "../data/"
OUT = "../data_processed/"
DT = 0.2
V_MAX = 15.0    # m/s
A_MAX = 6.0     # m/s^2
CELL = 0.2      # road grid cell size in metres

train = np.load(f"{OUT}train.npz")
val = np.load(f"{OUT}val.npz")
test = np.load(f"{OUT}test.npz")

rec_meta = pd.read_csv(f"{DATA}18_recordingMeta.csv")
image = plt.imread(f"{DATA}18_background.png")
metres_per_pixel = rec_meta["orthoPxToMeter"][0] * 12
height, width = image.shape[:2]
extent = [0, width * metres_per_pixel, -height * metres_per_pixel, 0]
```

`scipy.ndimage` has ready-made functions for working with images and grids. We use three of them below.

---

### Cell 2 · Road grid: where did cars drive?

```python
# every position of every train and val car, as one long list of (x, y)
points = np.concatenate([
    train["hist_map"].reshape(-1, 2), train["fut_map"].reshape(-1, 2),
    val["hist_map"].reshape(-1, 2), val["fut_map"].reshape(-1, 2),
])

# a grid that covers all points with 10 m extra on each side
x0, y0 = points.min(axis=0) - 10
x1, y1 = points.max(axis=0) + 10
n_cols = int((x1 - x0) / CELL) + 1
n_rows = int((y1 - y0) / CELL) + 1
road = np.zeros((n_rows, n_cols), dtype=bool)    # all False = no road yet

# mark the cell of every point as road
cols = ((points[:, 0] - x0) / CELL).astype(int)
rows = ((points[:, 1] - y0) / CELL).astype(int)
road[rows, cols] = True

print("grid size:", road.shape, "| road cells:", road.sum())
```

- `.reshape(-1, 2)` turns (N, 15, 2) into one long list of points (N·15, 2). `-1` means "work out this number yourself".
- A point at x lies in column `(x - x0) / 0.2`, rounded down by `.astype(int)`.
- `road[rows, cols] = True` marks all those cells in one go.
- We use **train and val** cars only. The test cars are never used to build anything.

---

### Cell 3 · Widen the road and fill holes

```python
road = ndimage.binary_dilation(road, iterations=8)   # grow the road by 8 cells (about 1.5 m); the final project uses 4 cells (0.8 m), see offline/03_road_and_rules.ipynb
road = ndimage.binary_fill_holes(road)               # fill small gaps inside

plt.figure(figsize=(9, 6))
plt.imshow(image, extent=extent)
road_layer = np.where(road, 1.0, np.nan)              # nan = not drawn, so only the road gets colour
plt.imshow(road_layer, extent=[x0, x0 + n_cols * CELL, y0, y0 + n_rows * CELL],
           origin="lower", alpha=0.4, cmap="Blues", vmin=0, vmax=1)
plt.xlim(extent[0], extent[1]); plt.ylim(extent[2], extent[3])
plt.title("Drivable area (blue)")
plt.savefig("../figures/03_drivable_area.png", dpi=150, bbox_inches="tight")
plt.show()
```

- We mark the **centre** of each car, so the real road is a bit wider. `binary_dilation` grows every True area by one cell per iteration.
- `np.nan` means "no value". matplotlib leaves those cells transparent.
- `origin="lower"`: row 0 of our grid is the smallest y, so it must be drawn at the bottom.
- **Check:** the blue area should cover the roads of the image, not the sidewalks.

---

### Cell 4 · Distance map

```python
outside = ndimage.distance_transform_edt(~road) * CELL   # for cells outside: metres to the nearest road cell
inside = ndimage.distance_transform_edt(road) * CELL     # for cells inside: metres to the nearest non-road cell
distance = np.where(road, -inside, outside)              # negative inside the road, positive outside

plt.figure(figsize=(8, 5))
plt.imshow(np.clip(distance, -6, 6), extent=[x0, x0 + n_cols * CELL, y0, y0 + n_rows * CELL],
           origin="lower", cmap="RdBu_r")
plt.colorbar(label="D(p) in metres")
plt.contour(distance, levels=[0], extent=[x0, x0 + n_cols * CELL, y0, y0 + n_rows * CELL],
            origin="lower", colors="black")
plt.title("Signed distance to the road edge (black line = edge)")
plt.savefig("../figures/03_distance_map.png", dpi=150, bbox_inches="tight")
plt.show()
```

- `distance_transform_edt` gives, for every True cell, the distance to the nearest False cell (in cells, so we multiply by 0.2 m).
- `~road` flips True and False.
- `np.where(condition, a, b)` takes `a` where the condition is True and `b` where it is False.
- Now the road rule is one lookup: **D(p) ≤ 0 means on the road.**

---

### Cell 5 · The three rules as code

```python
def road_distance(points):
    # D(p) for an array of points (..., 2): look up the cell the point falls in
    cols = np.clip(((points[..., 0] - x0) / CELL).astype(int), 0, n_cols - 1)
    rows = np.clip(((points[..., 1] - y0) / CELL).astype(int), 0, n_rows - 1)
    return distance[rows, cols]

def rule_values(fut_map, p0):
    # fut_map: (N, 30, 2) future points in the map, p0: (N, 2) last history point
    # returns the three rules as "g" values: g <= 0 means the rule holds
    p = np.concatenate([p0[:, None, :], fut_map], axis=1)            # (N, 31, 2): p0, p1, ..., p30
    g_road = road_distance(fut_map)                                    # (N, 30)
    speed = np.linalg.norm(p[:, 1:] - p[:, :-1], axis=-1) / DT        # (N, 30)
    acc = np.linalg.norm(p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2], axis=-1) / DT**2   # (N, 29)
    return g_road, speed - V_MAX, acc - A_MAX
```

| Rule | Formula | In code |
|---|---|---|
| road | `D(p_k) ≤ 0` | `road_distance(fut_map)` |
| speed | `‖p_k − p_(k−1)‖ / Δt − 15 ≤ 0` | distance between neighbour points / 0.2 |
| acceleration | `‖p_(k+1) − 2p_k + p_(k−1)‖ / Δt² − 6 ≤ 0` | change of the step / 0.04 |

- `p[:, 1:] - p[:, :-1]` = every point minus the one before it (all at once, no loop).
- `p0[:, None, :]` adds an extra axis so `p0` can be glued in front of the 30 future points.
- *Why acceleration:* if two steps in a row are very different, the car jumped or turned too sharply. This also covers sharp turns, so we do not need a separate turning rule.

---

### Cell 6 · Where do the limits come from?

```python
p0 = train["hist_map"][:, -1]
g_road, g_speed, g_acc = rule_values(train["fut_map"], p0)
speed = g_speed + V_MAX
acc = g_acc + A_MAX

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
axes[0].hist(speed.ravel(), bins=80)
axes[0].axvline(V_MAX, color="red")
axes[0].set_xlabel("speed (m/s)")
axes[0].set_title("Speed of real cars (red = limit)")
axes[1].hist(np.clip(acc.ravel(), 0, 10), bins=80)
axes[1].axvline(A_MAX, color="red")
axes[1].set_yscale("log")
axes[1].set_xlabel("acceleration (m/s², cut at 10)")
axes[1].set_title("Acceleration of real cars (log scale)")
plt.savefig("../figures/03_limits.png", dpi=150, bbox_inches="tight")
plt.show()

print("fastest real car:", round(speed.max(), 1), "m/s")
print("99.9% of accelerations are below:", round(np.percentile(acc, 99.9), 1), "m/s²")
```

- `.ravel()` makes one flat list of all values.
- The peak at 0 m/s = cars waiting at the junction.
- The limits must let real driving pass. 15 m/s is above the fastest car. 6 m/s² is above 99.9% of the data; the few higher values are drone tracking noise.

---

### Cell 7 · Check: does real test driving pass?

```python
p0 = test["hist_map"][:, -1]
g_road, g_speed, g_acc = rule_values(test["fut_map"], p0)

ok_road = (g_road <= 0).all(axis=1)       # True if all 30 points are on the road
ok_speed = (g_speed <= 0).all(axis=1)
ok_acc = (g_acc <= 0).all(axis=1)
ok_all = ok_road & ok_speed & ok_acc

print(f"road:         {ok_road.mean():.1%}")
print(f"speed:        {ok_speed.mean():.1%}")
print(f"acceleration: {ok_acc.mean():.1%}")
print(f"all three:    {ok_all.mean():.1%}")
```

- `.all(axis=1)`: True only if the rule holds at every point of the path.
- `.mean()` of True/False values = the share of True.
- **You should see** about 99% for all three. This is the **reference** for the results table: a strategy that reaches 99% is as good as real data.
- If road is much lower: widen the road more (`iterations=10`) and run cells 3-7 again.

---

### Cell 8 · Picture: a feasible and a broken path

```python
i = 165                                  # one test example (try others)
real = test["fut_map"][i]
broken = real.copy()
broken[15:21] += [6.0, 6.0]              # push 6 points 6 m to the side

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, path, title in [(axes[0], real, "real path"), (axes[1], broken, "broken path")]:
    g_road, g_speed, g_acc = rule_values(path[None], test["hist_map"][i:i + 1, -1])
    bad = (g_road[0] > 0) | (g_speed[0] > 0)
    bad[:-1] = bad[:-1] | (g_acc[0] > 0)   # acceleration is measured at points 1-29
    ax.imshow(image, extent=extent)
    ax.plot(test["hist_map"][i, :, 0], test["hist_map"][i, :, 1], "o-", color="tab:blue", markersize=3)
    ax.plot(path[:, 0], path[:, 1], "o-", color="tab:orange", markersize=3)
    ax.plot(path[bad, 0], path[bad, 1], "o", color="red", markersize=7)
    ax.set_xlim(path[:, 0].min() - 15, path[:, 0].max() + 15)
    ax.set_ylim(path[:, 1].min() - 15, path[:, 1].max() + 15)
    ax.set_title(f"{title}: {bad.sum()} of 30 points break a rule")
plt.savefig("../figures/03_feasible_vs_broken.png", dpi=150, bbox_inches="tight")
plt.show()
```

- `path[None]` adds a first axis, because `rule_values` expects many paths (N, 30, 2).
- `|` means "or" for True/False arrays.

---

### Cell 9 · Save the road grid

```python
np.savez(f"{OUT}road_grid.npz", road=road, distance=distance.astype(np.float32),
         x0=x0, y0=y0, cell=CELL)
print("saved")
```

---

### Done when

- [ ] about 99% of real test paths pass all three rules
- [ ] `data_processed/` has 4 files: `train.npz`, `val.npz`, `test.npz`, `road_grid.npz`
- [ ] you can explain: constraint g ≤ 0, feasible, signed distance D(p), why these limits

**Then:** upload the whole `data_processed/` folder to Drive: `Group 5 - Big Data/DM2583_project/data_processed/`.
Also upload `data/18_background.png` and `data/18_recordingMeta.csv` to `DM2583_project/raw/` (for pictures in Colab).

**Commit:** `git add offline/03_road_and_rules.ipynb figures/ && git commit -m "step 3: road and rules" && git push`

**For the report (Method, Constraints):** the rule table, the distance map figure, the limits figure, the 99% check.
