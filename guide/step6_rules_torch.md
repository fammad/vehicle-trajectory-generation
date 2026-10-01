# Step 6 · The rules in PyTorch and the repair function

**Where:** Colab, same notebook · **Time:** about 1.5 hours

**Goal:** the same 3 rules as in step 3, but in PyTorch, plus a function that **repairs** a broken path.
- Strategy B needs the *gradient* of the rules ("which way should the points move to break the rules less?"). PyTorch computes gradients for us, numpy can't.
- Strategies C and D need the repair function, called **projection Π**: it moves a path to the nearest path that keeps the rules.

Markdown cell: `## 4 · Rules and projection`

---

### Cell 1 · Road distance that has a gradient

```python
road_distance_grid = torch.tensor(road_file["distance"], device=device)
road_mask = road_file["road"]
X0, Y0, CELL = float(road_file["x0"]), float(road_file["y0"]), float(road_file["cell"])
N_ROWS, N_COLS = road_mask.shape

def road_distance(points):
    # D(p) for points (..., 2). Mixes the 4 nearest grid values (bilinear), so it changes smoothly.
    col = ((points[..., 0] - X0) / CELL).clamp(0, N_COLS - 1.001)
    row = ((points[..., 1] - Y0) / CELL).clamp(0, N_ROWS - 1.001)
    c0, r0 = col.floor().long(), row.floor().long()
    wc, wr = col - c0, row - r0                  # how far we are between grid nodes (0 to 1)
    top = (1 - wc) * road_distance_grid[r0, c0] + wc * road_distance_grid[r0, c0 + 1]
    bottom = (1 - wc) * road_distance_grid[r0 + 1, c0] + wc * road_distance_grid[r0 + 1, c0 + 1]
    return (1 - wr) * top + wr * bottom
```

- In step 3 we just looked up the cell. That jumps from one value to the next, so the gradient is zero almost everywhere.
- **Bilinear interpolation** mixes the 4 surrounding values by how close the point is. Now a tiny move of the point gives a tiny change of D, so the gradient exists.
- `.clamp(a, b)` keeps the number between a and b (so we never look outside the grid).

---

### Cell 2 · The three rules and the penalty

```python
def length(v):
    # length of 2D vectors. The tiny 1e-9 avoids a division by zero in the gradient at length 0.
    return torch.sqrt((v ** 2).sum(dim=-1) + 1e-9)

def rule_values(fut_map, p0):
    # fut_map: (N, 30, 2), p0: (N, 2). Returns g_road (N,30), g_speed (N,30), g_acc (N,29). g <= 0 means OK.
    p = torch.cat([p0[:, None, :], fut_map], dim=1)
    g_road = road_distance(fut_map)
    g_speed = length(p[:, 1:] - p[:, :-1]) / DT - V_MAX
    g_acc = length(p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]) / DT**2 - A_MAX
    return g_road, g_speed, g_acc

def is_feasible(fut_map, p0):
    g_road, g_speed, g_acc = rule_values(fut_map, p0)
    return (g_road <= TOL).all(dim=1) & (g_speed <= TOL).all(dim=1) & (g_acc <= TOL).all(dim=1)

def penalty(fut_map, p0):
    # P = sum of (how much a rule is broken)^2. It is 0 for a feasible path.
    g_road, g_speed, g_acc = rule_values(fut_map, p0)
    return (torch.relu(g_road) ** 2).sum(dim=1) + (torch.relu(g_speed) ** 2).sum(dim=1) + (torch.relu(g_acc) ** 2).sum(dim=1)
```

- Same formulas as step 3, written with `torch` instead of `np`.
- `torch.relu(g)` = `max(0, g)`: 0 when the rule holds, the amount of breaking when it doesn't.
- **Penalty P** adds up the squared breaking. Squared, so big breaks count much more than small ones.

**Check (same as step 3):**

```python
real = torch.tensor(test["fut_map"], device=device)
p0_all = torch.tensor(test["hist_map"][:, -1], device=device)
print(f"real test paths that keep all rules: {is_feasible(real, p0_all).float().mean():.1%}")
```

About 99% again.

---

### Cell 3 · Projection Π (the repair function)

```python
from scipy import ndimage

# for every cell of the grid: the row and column of the nearest road cell (computed once)
_, nearest = ndimage.distance_transform_edt(~road_mask, return_indices=True)
nearest_row = torch.tensor(nearest[0], device=device)
nearest_col = torch.tensor(nearest[1], device=device)

def fix_speed_and_acceleration(p, p0):
    # walk along the path from p0 and limit every step
    new_points = []
    before, last = None, p0
    for k in range(F):
        step = p[:, k] - last
        if before is not None:
            same_speed_step = last - before                  # the step if the car kept its velocity
            change = step - same_speed_step                  # how much the velocity changes
            limit = A_MAX * DT**2
            size = length(change)[:, None]
            change = torch.where(size > limit, change * limit / size, change)    # shorten if too big
            step = same_speed_step + change
        limit = V_MAX * DT
        size = length(step)[:, None]
        step = torch.where(size > limit, step * limit / size, step)              # shorten if too long
        new_point = last + step
        new_points.append(new_point)
        before, last = last, new_point
    return torch.stack(new_points, dim=1)

def fix_road(p):
    # move every point that is off the road to the nearest road cell
    col = ((p[..., 0] - X0) / CELL).round().long().clamp(0, N_COLS - 1)
    row = ((p[..., 1] - Y0) / CELL).round().long().clamp(0, N_ROWS - 1)
    off_road = road_distance(p) > 0
    new_x = X0 + nearest_col[row, col] * CELL
    new_y = Y0 + nearest_row[row, col] * CELL
    x = torch.where(off_road, new_x, p[..., 0])
    y = torch.where(off_road, new_y, p[..., 1])
    return torch.stack([x, y], dim=-1)

def project(fut_map, p0, repeats=3):
    p = fut_map
    for _ in range(repeats):
        p = fix_speed_and_acceleration(p, p0)
        p = fix_road(p)
    return p
```

- `torch.where(condition, a, b)`: like `np.where`. Here: if the step is too big, use the shortened step, otherwise keep it.
- **Speed and acceleration:** we walk from `p0`. If the velocity changes too much, we cut the change to exactly `A_MAX`. If a step is too long, we cut it to `V_MAX · Δt`.
- **Road:** `distance_transform_edt(..., return_indices=True)` also tells, for every cell, *which* road cell is nearest. Off-road points jump there.
- We repeat 3 times because fixing one rule can break another.
- The road fix comes last, so **the road rule always holds** after projection. Moving a point to the road edge can create a sharp bend, so the acceleration rule can still break there. This is a real effect of having several rules at once. Note it for the report.

---

### Cell 4 · Picture: before and after repair

```python
i = 165
c, origin, yaw, p0 = tensors_for(i, 20)
before = to_map(sample_plain(model, c), origin, yaw)
after = project(before, p0)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, paths, title in [(axes[0], before, "generated"), (axes[1], after, "after projection Π")]:
    ok = is_feasible(paths, p0).cpu().numpy()
    paths = paths.cpu().numpy()
    draw_map(ax)
    for k in range(len(paths)):
        ax.plot(paths[k, :, 0], paths[k, :, 1], color="orange" if ok[k] else "red", linewidth=1)
    ax.set_xlim(15, 75); ax.set_ylim(-55, 0)
    ax.set_title(f"{title}: {ok.mean():.0%} of paths keep all rules (red = breaks a rule)")
plt.savefig(OUTPUTS + "projection.png", dpi=150, bbox_inches="tight")
plt.show()
```

**For the report (Method, Constraints):** the penalty formula, the projection in 3 sentences, the remark that road comes last. **Figure:** `projection.png`.
