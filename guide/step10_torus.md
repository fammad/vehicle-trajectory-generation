# Step 10 · The torus (toy problem)

**Where:** Colab, same notebook, new part at the end · **Time:** about 2 hours

**Goal:** the same 4 strategies on a simple problem where we can *see* everything in 3D. This was Stage 1 of the course.
It also gives a nice comparison for the report: the torus has an **equality** rule (exactly on a surface), the cars have **inequality** rules (inside a region).

> If the professor gave a specific torus setup (radii, data distribution, sample count), use those numbers in Cell 1 instead of ours.

**The torus in plain words:** a donut. `R` = distance from the centre of the donut to the centre of the tube, `r` = radius of the tube.
A point is described by two angles: `θ` (around the big circle) and `φ` (around the tube, the *poloidal* angle).

**The rule:** distance to the surface `d(x) = sqrt((sqrt(x² + y²) − R)² + z²) − r` must be 0. That is `h(x) = 0`, an equality.

Markdown cell: `# Part 2 · Torus` and the 3 paragraphs above, short.

---

### Cell 1 · Make the data

```python
R, r = 2.0, 0.7
N_TORUS = 6000
rng = np.random.default_rng(0)

theta = rng.uniform(0, 2 * np.pi, N_TORUS)          # around the big circle: uniform
phi = rng.vonmises(mu=0.0, kappa=2.0, size=N_TORUS)  # around the tube: more points on the outside

def torus_points(theta, phi):
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.stack([x, y, z], axis=1)

torus_data = torus_points(theta, phi).astype(np.float32)
torus_train, torus_test = torus_data[:5000], torus_data[5000:]

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(projection="3d")
ax.scatter(*torus_train[:2000].T, s=2)
ax.set_title("torus training data")
plt.show()
```

- `vonmises` is a "normal distribution on a circle". With `kappa=2` most points are on the outer side of the tube (φ near 0). That makes the poloidal metric meaningful: a good model must copy this unevenness.
- `*torus_train.T` splits the (N, 3) array into x, y, z for the scatter plot.

---

### Cell 2 · Rule, penalty, projection

```python
def torus_distance(p):
    # signed distance to the torus surface. p: (N, 3) tensor
    rho = torch.sqrt(p[:, 0]**2 + p[:, 1]**2 + 1e-9)       # distance from the z axis
    return torch.sqrt((rho - R)**2 + p[:, 2]**2 + 1e-9) - r

def torus_penalty(p):
    return torus_distance(p) ** 2                          # equality rule: any distance is bad

def torus_project(p):
    # nearest point on the surface: go to the centre circle of the tube, then r outwards towards p
    rho = torch.sqrt(p[:, 0]**2 + p[:, 1]**2 + 1e-9)
    centre = torch.stack([R * p[:, 0] / rho, R * p[:, 1] / rho, torch.zeros_like(rho)], dim=1)
    direction = p - centre
    return centre + r * direction / torch.sqrt((direction**2).sum(dim=1, keepdim=True) + 1e-9)

def poloidal_angle(p):
    rho = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
    return np.arctan2(p[:, 2], rho - R)
```

- Unlike the road, projection onto a torus has an exact formula. And it moves **every** point (a point is almost never exactly on a surface), while the road projection only moves points outside the road.
- `poloidal_angle` gives φ back from a point. We use it to check the distribution around the tube.

---

### Cell 3 · Model and training

```python
class TorusModel(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 + 16, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x_t, t):
        freqs = torch.arange(1, 9, device=t.device) * np.pi
        t_features = torch.cat([torch.sin(t[:, None] * freqs), torch.cos(t[:, None] * freqs)], dim=1)
        return self.net(torch.cat([x_t, t_features], dim=1))

torch.manual_seed(0)
torus_model = TorusModel().to(device)
optimizer = torch.optim.Adam(torus_model.parameters(), lr=1e-3)
x_torus = torch.tensor(torus_train, device=device)
losses = []
for it in range(4000):
    x1 = x_torus[torch.randint(0, len(x_torus), (512,), device=device)]
    x0 = torch.randn_like(x1)
    t = torch.rand(len(x1), device=device)
    x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
    loss = ((torus_model(x_t, t) - (x1 - x0)) ** 2).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    losses.append(loss.item())
plt.plot(losses); plt.title("torus training loss"); plt.show()
```

Same Flow Matching as for the cars, but simpler: no history (no condition), 3 numbers instead of 60, no normalisation (the torus already lives around 0 with size about 2).

---

### Cell 4 · The four strategies on the torus

```python
def torus_sample(strategy="A", lam=1.0, n=1000, steps=50, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n, 3, device=device)
    for k in range(steps):
        t = k / steps
        with torch.no_grad():
            v = torus_model(x, torch.full((n,), t, device=device))
        x1_guess = x + (1 - t) * v
        x0_guess = x - t * v
        if strategy == "B":
            x1_guess = x1_guess.detach().requires_grad_(True)
            grad = torch.autograd.grad(torus_penalty(x1_guess).sum(), x1_guess)[0]
            v = v - lam * grad
        if strategy == "C":
            repaired = torus_project(x1_guess)
            x = (1 - t) * x0_guess + t * repaired
            v = repaired - x0_guess
        x = x + (1 / steps) * v
    x = x.detach()
    if strategy == "D":
        x = torus_project(x)
    return x
```

Exactly the same structure as `sample` for the cars. Only the rule and the projection changed.

---

### Cell 5 · The four torus metrics

```python
def torus_evaluate(p):
    d = torus_distance(p).abs().cpu().numpy()
    gen = p.cpu().numpy()
    real = torus_test[:len(gen)]
    swd = sliced_wasserstein(gen, real) - sliced_wasserstein(torus_train[:len(gen)], real)
    return {
        "mean constraint violation": d.mean(),
        "on-surface rate": (d < 0.01).mean(),
        "excess SWD": swd,
        "poloidal W1": wasserstein_distance(poloidal_angle(gen), poloidal_angle(real)),
    }

torus_rows = []
for strategy, lam in [("A", 0), ("B", 1.0), ("B", 10.0), ("C", 0), ("D", 0)]:
    for seed in [0, 1, 2]:
        result = torus_evaluate(torus_sample(strategy, lam, seed=seed))
        result["strategy"] = strategy if strategy != "B" else f"B λ={lam}"
        torus_rows.append(result)

torus_results = pd.DataFrame(torus_rows)
torus_results.to_csv(OUTPUTS + "torus_results.csv", index=False)
torus_results.groupby("strategy", sort=False).agg(["mean", "std"]).round(4)
```

- **Mean constraint violation:** average distance to the surface.
- **On-surface rate:** share of points closer than 0.01 to the surface.
- **Excess SWD:** same idea as for cars, in 3D.
- **Poloidal W1:** compares the φ distribution with real data. A strategy can put every point on the surface and still get this wrong (for example, pile points on the inside of the tube).
- Here the 3 seeds only change the sampling noise (one model). That's enough for the toy problem.

---

### Cell 6 · Pictures

```python
fig = plt.figure(figsize=(20, 4.5))
for k, (strategy, lam) in enumerate([("A", 0), ("B", 1.0), ("C", 0), ("D", 0)]):
    p = torus_sample(strategy, lam).cpu().numpy()
    ax = fig.add_subplot(1, 4, k + 1, projection="3d")
    ax.scatter(*p.T, s=2)
    ax.set_zlim(-2, 2)
    ax.set_title(strategy)
plt.savefig(OUTPUTS + "torus_samples.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(7, 4))
plt.hist(poloidal_angle(torus_test), bins=40, density=True, histtype="step", color="black", label="real")
for strategy in ["A", "C", "D"]:
    plt.hist(poloidal_angle(torus_sample(strategy).cpu().numpy()), bins=40, density=True, histtype="step", label=strategy)
plt.xlabel("poloidal angle φ (rad)")
plt.legend()
plt.title("Distribution around the tube")
plt.savefig(OUTPUTS + "torus_poloidal.png", dpi=150, bbox_inches="tight")
plt.show()
```

**What you will probably see:** A is close to the surface but not on it. D puts every point exactly on the surface and keeps the φ distribution. C also lands on the surface, but many points pile up on the **inner side** of the tube (big poloidal W1). In the φ histogram, A is hidden behind D: D only moves A's points straight onto the surface, so their angles stay the same.
Why: at the start of sampling the clean guess `x̂1` is poor and lies near the centre of the donut hole. Its nearest surface point is on the inner side, and C keeps pulling towards it in the early steps.
This is a real finding, not a bug, and good material for the discussion: repeated projection is only as good as the guess you project. With cars the history makes the early guess much better, so the effect is smaller.

**For the report:** one subsection "Toy problem: torus" (about 1 page): the table, these 2 figures, and the comparison equality vs inequality.
