# Step 7 · The four strategies

**Where:** Colab, same notebook · **Time:** about 1.5 hours

**Goal:** one sampling function that can run strategy A, B, C or D, with the **same trained model**.
The professor's rule: *the difference must come from how the constraint enters sampling, not from training different models.*

Markdown cell: `## 5 · Four strategies` and this table:

| | Strategy | At every one of the 50 steps |
|---|---|---|
| A | Unconstrained | nothing extra |
| B | Soft | push the direction a little away from breaking rules (strength λ) |
| C | Repeated hard | repair the current guess of the final path with Π, continue from the repaired path |
| D | Terminal | nothing; repair once at the very end |

**One new idea: the "clean guess".** At step k we are at time `t = k/50` on the way from noise to path.
If we jumped straight to the end now, we would land at `x̂1 = x_t + (1 − t)·v` (v = the network's direction).
And where we came from is `x̂0 = x_t − t·v`. B and C use `x̂1` to check the rules *during* sampling.

---

### Cell 1 · The sampling function

```python
def to_veh_steps(p_map, origin, yaw):
    # the opposite of to_map: map positions (N, 30, 2) -> normalised steps (N, 60)
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    dx = p_map[..., 0] - origin[:, None, 0]
    dy = p_map[..., 1] - origin[:, None, 1]
    pos = torch.stack([c * dx + s * dy, -s * dx + c * dy], dim=-1)
    steps = torch.diff(pos, dim=1, prepend=torch.zeros_like(pos[:, :1]))
    return (steps.reshape(-1, 60) - X_MEAN_T) / X_STD_T

def sample(model, c, origin, yaw, p0, strategy="A", lam=1.0, steps=50, seed=0, keep_frames=False):
    torch.manual_seed(seed)
    x = torch.randn(len(c), 60, device=device)
    frames = []
    for k in range(steps):
        t = k / steps
        with torch.no_grad():
            v = model(x, torch.full((len(c),), t, device=device), c)
        x1_guess = x + (1 - t) * v           # clean guess: where we land if we jump to the end
        x0_guess = x - t * v                 # where we came from

        if strategy == "B":
            # gradient of the penalty: which way makes the rules break more. We go the other way.
            x1_guess = x1_guess.detach().requires_grad_(True)
            P = penalty(to_map(x1_guess, origin, yaw), p0).sum()
            grad = torch.autograd.grad(P, x1_guess)[0]
            size = grad.norm(dim=1, keepdim=True)
            grad = torch.where(size > 1, grad / size, grad)      # cap the push at length 1
            v = v - lam * grad

        if strategy == "C":
            repaired = to_veh_steps(project(to_map(x1_guess, origin, yaw), p0), origin, yaw)
            x = (1 - t) * x0_guess + t * repaired                 # move the current point onto the repaired line
            v = repaired - x0_guess                               # and head for the repaired path

        x = x + (1 / steps) * v
        if keep_frames:
            frames.append(to_map(x.detach(), origin, yaw).cpu().numpy())

    with torch.no_grad():
        paths = to_map(x, origin, yaw)
        if strategy == "D":
            paths = project(paths, p0)
    if keep_frames:
        return paths, frames
    return paths
```

- **A:** only the Euler step, like `sample_plain` in step 5.
- **B:** `requires_grad_(True)` tells PyTorch to track `x1_guess`, `torch.autograd.grad` gives ∂P/∂x1_guess. Subtracting it moves the direction away from breaking. The cap stops one huge violation from throwing the path away. **λ** = how hard we push.
- **C:** repair the guess, then *rebuild* the current point on the straight line between `x0_guess` and the repaired path. This is the Christopher et al. (2024) idea moved from diffusion to Flow Matching.
- **D:** normal sampling, one repair at the end.
- `keep_frames=True` stores the path after every step. We use it for the animation in step 11.

---

### Cell 2 · Quick test on 500 test histories

```python
n = 500
c = c_te[:n]
origin = torch.tensor(test["origin"][:n], device=device)
yaw = torch.tensor(test["yaw"][:n], device=device)
p0 = torch.tensor(test["hist_map"][:n, -1], device=device)

for strategy in ["A", "B", "C", "D"]:
    paths = sample(model, c, origin, yaw, p0, strategy=strategy, lam=1.0)
    print(f"{strategy}: {is_feasible(paths, p0).float().mean():.1%} of paths keep all rules")
```

**You should see** roughly: A lowest, B higher, C close to 99%, D in between (road is fixed but bends break acceleration).

---

### Cell 3 · The demo figure

```python
def demo_figure(i, n=20, lam=1.0, save=None):
    c, origin, yaw, p0 = tensors_for(i, n)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    for ax, strategy in zip(axes, ["A", "B", "C", "D"]):
        paths = sample(model, c, origin, yaw, p0, strategy=strategy, lam=lam, seed=1)
        ok = is_feasible(paths, p0).cpu().numpy()
        paths = paths.cpu().numpy()
        draw_map(ax)
        ax.plot(test["hist_map"][i, :, 0], test["hist_map"][i, :, 1], "o-", color="tab:blue", markersize=3)
        for k in range(n):
            ax.plot(paths[k, :, 0], paths[k, :, 1], color="orange" if ok[k] else "red", linewidth=1)
        centre = test["hist_map"][i, -1]
        ax.set_xlim(centre[0] - 30, centre[0] + 30)
        ax.set_ylim(centre[1] - 30, centre[1] + 30)
        ax.set_title(f"{strategy}: {ok.mean():.0%} keep all rules")
    plt.suptitle("same history, 20 generated futures per strategy (red = breaks a rule)")
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.show()

demo_figure(165, save=OUTPUTS + "demo_165.png")
```

Try other examples: `demo_figure(40)`, `demo_figure(900)`, ... Keep 2-3 that show the differences well, for the report and the talk.

**For the report (Method):** the strategy table and the clean-guess formula. **Figure:** the demo figure. This is the most important picture of the project.
