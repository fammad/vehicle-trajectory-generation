# Build plan: Constrained Flow Matching for vehicle trajectories

Group 5 · DM2583 · Deadlines: checkpoint 13 Oct · code + demo 14 Oct · report + talk 15 Oct

---

## The project in 5 sentences

1. We have drone videos of one intersection (inD, Frankenburg) where every car's position is known 25 times per second.
2. We cut the car tracks into small pieces: **3 s of history** (what the car did) and **6 s of future** (what it did next).
3. We train a **Flow Matching** model that looks at the history and draws a possible future.
4. Some drawn futures are impossible: off the road, too fast, too jerky. We write these as **3 rules**.
5. We compare **4 ways (A, B, C, D)** of making the model follow the rules, and measure what each way costs.

---

## How we work

- **Offline (laptop):** steps 1-3. They need the big raw CSV files (1.2 GB). Output: one small folder `data_processed/` (about 8 MB).
- **Online (Colab):** steps 4-9 in **one notebook**. Upload `data_processed/` to Google Drive once, then everyone works in the same notebook.
- **One notebook per offline step.** Every step ends with **a picture** and **a check**, so we can see it worked and show it.
- **Simple code:** only `numpy`, `pandas`, `matplotlib`, one function from `scipy`, and `torch` (Colab only). No classes except the model. Short functions with comments. If simple is good enough, simple wins.
- **Git:** code and notebooks go to GitHub. Data never goes to GitHub (see `.gitignore`).

The old version of everything is in `_old_version/` (not in git). Use it only to peek if stuck.

---

## Folder layout (what it will look like at the end)

```
Project-vehicle-trajectory-generation/
├── BUILD_PLAN.md                 this file
├── README.md                     how to set up and run (short)
├── requirements.txt
├── data/                         raw inD files (not in git)
├── maps/                         official maps (reference only)
├── offline/
│   ├── 01_explore_data.ipynb     step 1
│   ├── 02_make_samples.ipynb     step 2
│   └── 03_road_and_rules.ipynb   step 3
├── data_processed/               output of steps 2-3 (not in git, goes to Drive)
├── figures/                      pictures saved by the notebooks (for report and slides)
└── colab/
    └── DM2583_project.ipynb      steps 4-9 (the shared notebook)
```

---

## Step 0 · Setup (once, 20 min)

**What:** install Python and the libraries in a separate box (a *virtual environment*) so nothing breaks on the laptop.

```bash
cd ~/Desktop/"DM2583 - Big Data"/Project-vehicle-trajectory-generation
python3.11 -m venv .venv          # make the box (if python3.11 is missing: brew install python@3.11)
source .venv/bin/activate         # step into the box (do this every time you open a new terminal)
pip install numpy pandas matplotlib scipy jupyter
jupyter notebook                  # opens the browser
```

**Check:** a new notebook runs `import numpy, pandas, matplotlib, scipy` without errors.

---

## OFFLINE

### Step 1 · Explore the data → `offline/01_explore_data.ipynb`

**Goal:** understand what is inside the files before changing anything.

**What we do**
1. Open one recording (18) with pandas: `18_tracks.csv`, `18_tracksMeta.csv`, `18_recordingMeta.csv`.
2. Look at the columns: `trackId`, `frame`, `xCenter`, `yCenter`, `heading`, `xVelocity`, ...
3. Count road users by class (car, pedestrian, bicycle, truck_bus).
4. Draw all car tracks on top of the aerial image `18_background.png`.
5. Draw one car's speed over time.

**Pictures:** tracks on the aerial image · speed of one car over time.
**Check:** the tracks lie on the roads in the picture (then our coordinates are right).
**Output:** `figures/01_tracks_on_map.png`

**Words**
- *Recording*: one drone video (about 18 min). We use 12 recordings (18 to 29), all at the same intersection.
- *Track*: the full path of one road user in one recording.
- *Frame*: one moment in the video. 25 frames per second.

---

### Step 2 · Make the samples → `offline/02_make_samples.ipynb`

**Goal:** turn raw tracks into training examples the model can learn from.

**What we do** (first on recording 18 so we can look at each step, then a loop over all 12)
1. **Filter:** keep only `car` and `truck_bus`.
2. **Downsample:** keep every 5th frame. 25 Hz → 5 Hz, so one point every 0.2 s.
3. **Clean:** drop parked cars (moved less than 2 m) and tracks shorter than 9 s.
4. **Cut windows:** slide along each track and cut 45 points: 15 history (3 s) + 30 future (6 s). Start a new window every 5 points (1 s).
5. **Vehicle frame:** move each window so the car's last history point is at (0, 0) and rotate it so the car faces right (+x). Keep the original position (`origin`) and angle (`yaw`) to turn it back later.
6. **Split by recording:** train = 18-25, val = 26-27, test = 28-29.
7. **Save** `train.npz`, `val.npz`, `test.npz` in `data_processed/`.

**Pictures:** kept vs parked cars on the map · 4 overlapping windows from one car · one window in map frame next to the same window in vehicle frame.
**Check:** about 1,900 cars and 14,500 windows. Turning a window to the vehicle frame and back gives the same numbers.
**Output:** `data_processed/train.npz`, `val.npz`, `test.npz`

**Words**
- *Window*: one training example = 3 s history + 6 s future of one car.
- *Sample*: the 6 s future = 30 points × (x, y) = **60 numbers**. This is what the model generates.
- *Condition*: the 3 s history. The model gets it as input, it never generates it.
- *Vehicle frame*: coordinates seen from the car. Makes every window start the same way, so the model learns the *shape* of driving, not the *place*.
- *Split by recording*: windows of one car overlap a lot. If one went to train and the next to test, the test would be too easy. So whole recordings go to one split.

---

### Step 3 · Road and rules → `offline/03_road_and_rules.ipynb`

**Goal:** write the 3 rules as numbers we can check, and check that real driving passes them.

**What we do**
1. **Road grid:** make an empty grid over the intersection with cells of 0.2 m. Mark every cell a train or val car drove through. Widen it by 1.5 m and fill small holes. That is the *drivable area*.
2. **Distance map:** for every cell, the distance to the road edge. Negative inside the road, positive outside. One `scipy` function does this (`distance_transform_edt`).
3. **The 3 rules** for a path `p_0 … p_30` (`p_0` = last history point, Δt = 0.2 s):

   | Rule | Formula | Means |
   |---|---|---|
   | road | `D(p_k) ≤ 0` | every point on the road |
   | speed | `‖p_k − p_(k−1)‖ / Δt ≤ 15 m/s` | not faster than 15 m/s |
   | acceleration | `‖p_(k+1) − 2p_k + p_(k−1)‖ / Δt² ≤ 6 m/s²` | no jumps or very sharp turns |

4. **Check real test paths** against the rules.
5. Save `road_grid.npz`.

**Pictures:** drivable area on the aerial image · distance map (blue inside, red outside) · histograms of speed and acceleration with the limits as red lines · one real path (all green) next to a broken path (red points).
**Check:** about 99% of real test paths pass all 3 rules. (The 1% are tracking noise. This 99% is our "real data" reference.)
**Output:** `data_processed/road_grid.npz` → **upload the whole `data_processed/` folder to Drive.**

**Words**
- *Constraint / rule*: a check that says yes or no for every generated path. Written as `g ≤ 0`.
- *Feasible*: a path that passes all rules at every point.
- *Signed distance D(p)*: how far point p is from the road edge, with a minus sign inside.

---

## ONLINE (Colab, one notebook `colab/DM2583_project.ipynb`)

Every section starts with a short text cell: what, why, words. Then code, then a picture.

### Step 4 · Load and look (section 1-2)
Connect Drive, load the 4 `.npz` files, draw 20 random training futures and one example.
**Check:** same counts as offline.

### Step 5 · Flow Matching model (section 3)
**Idea:** start from random noise and move it step by step to a realistic path. A small neural network learns which direction to move.
- Training pair: noise `x0`, real future `x1`, random time `t` between 0 and 1.
- Mixed point: `x_t = (1 − t)·x0 + t·x1`. Target direction: `x1 − x0`.
- Loss: how far the network's direction is from the target (mean squared error).
- Model: MLP (a few layers of `Linear` + activation), input = mixed point + history + time.
- We let the model generate the 30 **steps** of the path (movement per 0.2 s) and add them up to positions. This keeps paths smooth.

**Pictures:** training loss going down · 20 generated futures next to the real one for the same history.
**Check:** generated paths look like driving and follow the road in most cases.

### Step 6 · The rules in PyTorch + projection (section 4)
Same 3 rules as step 3, written in `torch` so strategy B can use gradients.
**Projection Π:** a function that repairs a path: first limit speed and acceleration point by point, then move off-road points to the nearest road cell.
**Pictures:** a broken path before and after projection.

### Step 7 · The 4 strategies (section 5)
One `sample()` function, 50 small steps from noise to path. Same trained model for all four.

| | Name | At every step |
|---|---|---|
| A | Unconstrained | nothing extra |
| B | Soft | gently push away from breaking rules (strength λ) |
| C | Repeated hard | repair the current estimate with Π, continue from the repaired path |
| D | Terminal | nothing, repair once at the end |

**Picture:** the demo figure: same history, 20 paths per strategy, broken points red.

### Step 8 · Metrics (section 6)
- **Feasibility:** how much each rule is broken (violation), share of paths with no broken rule (on-surface rate).
- **Fidelity:** how close generated paths are to real paths (excess SWD), how far points are from the road edge (road-edge W1), how spread out 20 paths for one history are (diversity), left/straight/right shares, jerk (kinks).
- Real test data goes through the same metrics = reference row.

### Step 9 · Experiment and figures (section 7)
3 seeds × (A, B with 3 λ values, C, D). Table with mean ± std. Trade-off plot: feasibility vs fidelity.
**Output:** `outputs/results.csv`, `outputs/tradeoff.png`, `outputs/demo.png` → straight into the report and slides.

### Optional · Torus
Same 4 strategies on the toy torus problem. Only if time is left.

---

## Order and rough time

| Step | Where | Time |
|---|---|---|
| 0 Setup | laptop | 20 min |
| 1 Explore | laptop | 1 h |
| 2 Samples | laptop | 2 h |
| 3 Road + rules | laptop | 1.5 h |
| upload `data_processed/` to Drive | | 5 min |
| 4 Load + look | Colab | 30 min |
| 5 Model | Colab | 2 h |
| 6 Rules in torch + projection | Colab | 1.5 h |
| 7 Strategies | Colab | 1.5 h |
| 8 Metrics | Colab | 1.5 h |
| 9 Experiment + figures | Colab | 1 h + run time |
| Report + slides | | rest |

After each step: commit to git with a short message, e.g. `git add offline/01_explore_data.ipynb && git commit -m "step 1: explore data"`.
