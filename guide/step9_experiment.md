# Step 9 · The experiment and the result figures

**Where:** Colab, same notebook · **Time:** 1 hour + 10-20 min of GPU time

**Goal:** the numbers and figures that go into the Results section.
- 3 seeds (0, 1, 2): 3 trained models. Same code, different randomness. Shows how stable the results are.
- For every model: A, B with 3 values of λ, C, D. Same 2,000 test histories.
- Mean ± standard deviation over the 3 seeds.

Markdown cell: `## 7 · Experiment`

---

### Cell 1 · Choose λ on the validation set

```python
n = 500
c_v = c_va[:n]
origin_v = torch.tensor(val["origin"][:n], device=device)
yaw_v = torch.tensor(val["yaw"][:n], device=device)
p0_v = torch.tensor(val["hist_map"][:n, -1], device=device)

for lam in [0.1, 0.3, 1.0, 3.0, 10.0]:
    paths = sample(model, c_v, origin_v, yaw_v, p0_v, strategy="B", lam=lam)
    print(f"λ = {lam}: {is_feasible(paths, p0_v).float().mean():.1%} keep all rules")
```

Pick 3 values: a weak one, a middle one and a strong one (for example 0.3, 1, 3). Write them in `LAMBDAS` below.
We choose on **val**, never on test, so the test numbers stay fair.

---

### Cell 2 · Run everything

```python
SEEDS = [0, 1, 2]
LAMBDAS = [0.3, 1.0, 3.0]
RUNS = [("A", 0.0)] + [("B", lam) for lam in LAMBDAS] + [("C", 0.0), ("D", 0.0)]

rows = []
for seed in SEEDS:
    model_s = get_model(seed)                 # trains if not saved yet
    for strategy, lam in RUNS:
        paths = sample(model_s, c, origin, yaw, p0, strategy=strategy, lam=lam, seed=seed)
        result = evaluate(paths, real, p0, origin, yaw)
        result["diversity (m)"] = diversity(model_s, strategy, lam)
        result["strategy"] = strategy if strategy != "B" else f"B λ={lam}"
        result["seed"] = seed
        rows.append(result)
        print(f"seed {seed} | {result['strategy']:9s} | on-surface {result['on-surface rate']:.1%}")

results = pd.DataFrame(rows)
results.to_csv(OUTPUTS + "results.csv", index=False)
```

- `[("B", lam) for lam in LAMBDAS]` is a *list comprehension*: a short way to make a list with a loop.
- A `DataFrame` made from a list of dictionaries: every dictionary becomes a row.
- `results.csv` is saved to Drive. If Colab disconnects later you can load it with `pd.read_csv` instead of running again.

---

### Cell 3 · The results table

```python
order = list(results["strategy"].unique())
mean = results.groupby("strategy").mean(numeric_only=True).drop(columns="seed").loc[order]
std = results.groupby("strategy").std(numeric_only=True).drop(columns="seed").loc[order]

table = mean.round(3).astype(str) + " ± " + std.fillna(0).round(3).astype(str)
table.loc["real data"] = pd.Series(reference).round(3).astype(str)
table.T
```

- `groupby("strategy")` groups the 3 seeds of each strategy; `.mean()` and `.std()` give mean and spread.
- `.T` flips the table so strategies are columns. Copy it into the report (or use `table.T.to_latex()` for LaTeX).

---

### Cell 4 · Figure: feasibility vs fidelity

```python
fig, ax = plt.subplots(figsize=(6.5, 5))
for name in order:
    ax.errorbar(mean.loc[name, "excess SWD"], mean.loc[name, "on-surface rate"],
                xerr=std.loc[name, "excess SWD"], yerr=std.loc[name, "on-surface rate"], fmt="o", capsize=3)
    ax.annotate(name, (mean.loc[name, "excess SWD"], mean.loc[name, "on-surface rate"]),
                textcoords="offset points", xytext=(6, 4))
ax.axhline(reference["on-surface rate"], color="gray", linestyle="--", label="real data")
ax.set_xlabel("excess SWD  (left = more realistic)")
ax.set_ylabel("on-surface rate  (up = more rules kept)")
ax.legend()
ax.set_title("Feasibility vs fidelity")
plt.savefig(OUTPUTS + "tradeoff.png", dpi=150, bbox_inches="tight")
plt.show()
```

The best place is **top left**. Error bars = spread over the 3 seeds.

---

### Cell 5 · Figure: diversity and kinks

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(order, mean["diversity (m)"], yerr=std["diversity (m)"], capsize=3)
axes[0].set_ylabel("spread of 20 end points (m)")
axes[0].set_title("Diversity")
axes[1].bar(order, mean["jerk p95 (m/s³)"], yerr=std["jerk p95 (m/s³)"], capsize=3)
axes[1].axhline(reference["jerk p95 (m/s³)"], color="gray", linestyle="--", label="real data")
axes[1].set_ylabel("jerk, 95th percentile (m/s³)")
axes[1].set_title("Kinks")
axes[1].legend()
for ax in axes:
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(OUTPUTS + "diversity_and_jerk.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

### What to look for (check against the hypothesis in the proposal)

- A: most diverse, most violations.
- B: more λ → more rules kept, less diversity. A smooth trade-off.
- C: almost all rules kept. Does diversity drop?
- D: road always kept, but higher jerk (kinks where the repair lands).

Write 3-5 sentences per point. Unexpected results are results too: say what you expected and what happened.

**For the report (Results):** the table, `tradeoff.png`, `diversity_and_jerk.png`, 2 demo figures, and the sentences above.
