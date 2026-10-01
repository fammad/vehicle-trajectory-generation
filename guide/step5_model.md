# Step 5 · Flow Matching model

**Where:** Colab, same notebook · **Time:** about 2 hours

**Goal:** train a network that, given 3 s of history, draws a possible 6 s future.

**The idea of Flow Matching, in plain words**
- Take a real future `x1` (60 numbers) and pure random noise `x0` (60 random numbers).
- Draw a straight line between them. A point on that line is `x_t = (1 − t)·x0 + t·x1`. At `t = 0` it is noise, at `t = 1` it is the real path.
- Along the line, the direction is always `x1 − x0`.
- The network sees `x_t`, `t` and the history and must guess that direction. That's all the training does.
- To **generate**, start from new noise and take 50 small steps in the direction the network says. You end up at a path that looks real.

Markdown cell: `## 3 · Flow Matching model` plus the 5 bullets above, short.

---

### Cell 1 · Prepare the numbers

```python
def to_steps(fut_veh):
    # positions (N, 30, 2) -> steps (N, 30, 2): how far the car moves in each 0.2 s
    start = np.zeros_like(fut_veh[:, :1])                         # the car starts at (0, 0)
    return np.diff(np.concatenate([start, fut_veh], axis=1), axis=1)

def flat(a):
    return a.reshape(len(a), -1)       # (N, 30, 2) -> (N, 60)

x_train = flat(to_steps(train["fut_veh"]))   # what we generate: 60 numbers
c_train = flat(train["hist_veh"])            # the condition: 30 numbers

# mean and spread, from TRAIN only
X_MEAN, X_STD = x_train.mean(axis=0), x_train.std(axis=0) + 1e-6
C_MEAN, C_STD = c_train.mean(axis=0), c_train.std(axis=0) + 1e-6

def prepare(split):
    x = (flat(to_steps(split["fut_veh"])) - X_MEAN) / X_STD
    c = (flat(split["hist_veh"]) - C_MEAN) / C_STD
    return torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(c, dtype=torch.float32, device=device)

x_tr, c_tr = prepare(train)
x_va, c_va = prepare(val)
x_te, c_te = prepare(test)
print(x_tr.shape, c_tr.shape)
```

- **Steps instead of positions:** the model makes the 30 *steps* of the path. Adding them up gives the positions back (same 60 numbers, other form).
  Why: if every position gets its own small error, the path zig-zags and breaks the acceleration rule all the time. Small errors in steps add up smoothly.
- **Normalisation:** subtract the mean, divide by the spread. Then all numbers are around 0 with spread 1, like the noise. Networks learn much better like that.
  We compute mean and spread from **train only**, so val and test stay unseen.
- `+ 1e-6` avoids dividing by zero.
- `torch.tensor(..., device=device)` moves the data to the GPU.

---

### Cell 2 · The network

```python
class FlowModel(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(60 + 30 + 16, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 60),
        )

    def forward(self, x_t, t, c):
        # turn the single number t into 16 wave values, so the network can "read" time easily
        freqs = torch.arange(1, 9, device=t.device) * np.pi
        t_features = torch.cat([torch.sin(t[:, None] * freqs), torch.cos(t[:, None] * freqs)], dim=1)
        return self.net(torch.cat([x_t, c, t_features], dim=1))

model = FlowModel().to(device)
print("number of parameters:", sum(p.numel() for p in model.parameters()))
```

- An **MLP** (multi-layer perceptron) = layers of `Linear` (multiply by weights, add bias) with an activation (`SiLU`, a smooth version of "keep positive values") in between.
- Input: noisy path (60) + history (30) + time features (16) = 106 numbers. Output: a direction (60 numbers).
- `class FlowModel(nn.Module)` is how PyTorch wants a network to be written: build the layers in `__init__`, use them in `forward`.
- About 0.6 million parameters (the numbers the network learns).

---

### Cell 3 · Loss and training

```python
def fm_loss(model, x1, c):
    x0 = torch.randn_like(x1)                       # noise
    t = torch.rand(len(x1), device=device)          # a random time for every example
    x_t = (1 - t[:, None]) * x0 + t[:, None] * x1   # point on the straight line
    target = x1 - x0                                 # direction of the line
    prediction = model(x_t, t, c)
    return ((prediction - target) ** 2).mean()      # mean squared error

def train_model(seed, epochs=150, batch_size=256):
    torch.manual_seed(seed)
    model = FlowModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    val_losses = []
    for epoch in range(epochs):
        order = torch.randperm(len(x_tr), device=device)        # shuffle
        for start in range(0, len(x_tr), batch_size):
            batch = order[start:start + batch_size]
            loss = fm_loss(model, x_tr[batch], c_tr[batch])
            optimizer.zero_grad()    # forget old gradients
            loss.backward()          # compute new gradients
            optimizer.step()         # change the weights a little
        with torch.no_grad():
            val_losses.append(fm_loss(model, x_va, c_va).item())
        if epoch % 25 == 0:
            print(f"epoch {epoch}: val loss {val_losses[-1]:.3f}")
    return model, val_losses
```

- **Epoch** = one pass over all training examples. **Batch** = 256 examples at once.
- **Seed** = the start value for randomness. Same seed → same result. We will train with seeds 0, 1, 2 to see how much results change.
- `Adam` is the standard optimizer: it uses the gradients to change the weights so the loss goes down. `lr` = learning rate = step size.
- `torch.no_grad()`: we only measure the validation loss, no learning, so no gradients needed.
- The loss never reaches 0: from noise there are many possible futures (left, straight, wait), so the direction can't be guessed exactly. That's fine.

---

### Cell 4 · Train (or load) and save

```python
def get_model(seed):
    path = OUTPUTS + f"model_seed{seed}.pt"
    model = FlowModel().to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print("loaded", path)
    else:
        model, val_losses = train_model(seed)
        torch.save(model.state_dict(), path)
        plt.plot(val_losses)
        plt.xlabel("epoch"); plt.ylabel("validation loss"); plt.title(f"training, seed {seed}")
        plt.savefig(OUTPUTS + f"loss_seed{seed}.png", dpi=150, bbox_inches="tight")
        plt.show()
    model.eval()
    return model

model = get_model(0)
```

- Training takes 1-3 minutes on the GPU. The weights are saved, so next time (or for a teammate) it just loads.
- **If you change the model or the data, delete the `.pt` files in `outputs/`**, otherwise the old model is loaded.

**Picture:** the validation loss going down.

---

### Cell 5 · First samples

```python
X_MEAN_T = torch.tensor(X_MEAN, dtype=torch.float32, device=device)
X_STD_T = torch.tensor(X_STD, dtype=torch.float32, device=device)

def to_map(x, origin, yaw):
    # normalised steps (N, 60) -> positions on the map (N, 30, 2)
    steps = (x * X_STD_T + X_MEAN_T).reshape(-1, F, 2)
    pos = torch.cumsum(steps, dim=1)                     # add up the steps
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    map_x = c * pos[..., 0] - s * pos[..., 1] + origin[:, None, 0]
    map_y = s * pos[..., 0] + c * pos[..., 1] + origin[:, None, 1]
    return torch.stack([map_x, map_y], dim=-1)

def sample_plain(model, c, steps=50):
    x = torch.randn(len(c), 60, device=device)           # start: noise
    for k in range(steps):
        t = torch.full((len(c),), k / steps, device=device)
        with torch.no_grad():
            x = x + (1 / steps) * model(x, t, c)          # small step in the predicted direction
    return x

def tensors_for(i, n):
    # condition, origin, yaw and p0 of test example i, repeated n times
    c = c_te[i:i + 1].repeat(n, 1)
    origin = torch.tensor(test["origin"][i:i + 1], device=device).repeat(n, 1)
    yaw = torch.tensor(test["yaw"][i:i + 1], device=device).repeat(n)
    p0 = torch.tensor(test["hist_map"][i:i + 1, -1], device=device).repeat(n, 1)
    return c, origin, yaw, p0

i = 165
c, origin, yaw, p0 = tensors_for(i, 20)
paths = to_map(sample_plain(model, c), origin, yaw).cpu().numpy()

fig, ax = plt.subplots(figsize=(7, 6))
draw_map(ax)
ax.plot(test["hist_map"][i, :, 0], test["hist_map"][i, :, 1], "o-", color="tab:blue", markersize=3, label="history")
for path in paths:
    ax.plot(path[:, 0], path[:, 1], color="orange", linewidth=1, alpha=0.7)
ax.plot(test["fut_map"][i, :, 0], test["fut_map"][i, :, 1], "--", color="lime", linewidth=2, label="real future")
ax.set_xlim(15, 75); ax.set_ylim(-55, 0)
ax.legend()
ax.set_title("20 generated futures for the same history")
plt.savefig(OUTPUTS + "first_samples.png", dpi=150, bbox_inches="tight")
plt.show()
```

- `to_map` undoes everything: normalisation → steps → positions (`cumsum` = running sum) → rotate and move back to the map. We need map positions to check the rules.
- **Euler steps:** `x = x + (1/50) · direction`, 50 times. This is the simplest way to follow a flow.
- `.cpu().numpy()` brings the result back from the GPU to numpy for plotting.

**Check:** most orange paths follow roads. Some may not. That is exactly why we need the rules.

**For the report (Method):** the Flow Matching idea in 4 lines, the loss formula, the model size, steps-instead-of-positions. **Figure:** `first_samples.png`.
