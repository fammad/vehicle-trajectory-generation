# Step 4 · Colab: connect and load

**Where:** Colab · **Notebook:** `DM2583_project.ipynb` in Drive (`Group 5 - Big Data/DM2583_project/`) · **Time:** 30 min

From now on everything goes into **one notebook**. Steps 4 to 10 are its sections, in order.
Each section starts with a markdown cell (title + 2-3 lines of what and why), then the code cells below.

**Before you start**
- Drive folder `DM2583_project/` has `data_processed/` (4 files from step 3) and `raw/` (`18_background.png`, `18_recordingMeta.csv`).
- The shared folder is in your **My Drive** (right-click → Organize → Add shortcut).
- In Drive: New → More → Google Colaboratory. Name it `DM2583_project`.
- Runtime → Change runtime type → **T4 GPU**.

**Working together:** only one person edits the notebook at a time (say it in the group chat). Colab saves to Drive by itself.
When the notebook works, also download it (File → Download .ipynb) into `colab/` in the repo and commit it.

---

### Cell 1 · Connect Google Drive

Markdown: `# DM2583 · Group 5 · Constrained Flow Matching for vehicle trajectories` and `## 1 · Setup`

```python
from google.colab import drive
drive.mount("/content/drive")

PROJECT = "/content/drive/MyDrive/Group 5 - Big Data/DM2583_project/"
DATA = PROJECT + "data_processed/"
RAW = PROJECT + "raw/"
OUTPUTS = PROJECT + "outputs/"

import os
os.makedirs(OUTPUTS, exist_ok=True)
print(os.listdir(DATA))
```

- `drive.mount` asks for permission once, then your Drive is a folder at `/content/drive`.
- Everything we save (models, results, figures) goes to `outputs/`, so it stays in Drive after Colab closes.
- **You should see** the 4 files of `data_processed/`.

---

### Cell 2 · Imports and settings

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print("running on:", device)

DT = 0.2       # seconds between points
H = 15         # history points (3 s)
F = 30         # future points (6 s) -> one sample = 30 x 2 = 60 numbers
V_MAX = 15.0   # m/s
A_MAX = 6.0    # m/s^2
TOL = 1e-3     # tiny tolerance when we check rules (rounding errors)
```

- `torch` (PyTorch) is the deep learning library. `nn` has the building blocks of neural networks.
- `device`: "cuda" means the GPU. Everything big we put on the GPU, because it is much faster.

---

### Cell 3 · Load the data

```python
train = dict(np.load(DATA + "train.npz"))
val = dict(np.load(DATA + "val.npz"))
test = dict(np.load(DATA + "test.npz"))
road_file = np.load(DATA + "road_grid.npz")

print("train:", len(train["fut_veh"]), "| val:", len(val["fut_veh"]), "| test:", len(test["fut_veh"]))
```

`dict(...)` turns the loaded file into a normal dictionary, so `train["fut_veh"]` gives the array.
**Check:** same numbers as at the end of step 2.

---

### Cell 4 · The aerial image (for all pictures)

Markdown: `## 2 · Look at the data`

```python
rec_meta = pd.read_csv(RAW + "18_recordingMeta.csv")
image = plt.imread(RAW + "18_background.png")
metres_per_pixel = rec_meta["orthoPxToMeter"][0] * 12
height, width = image.shape[:2]
EXTENT = [0, width * metres_per_pixel, -height * metres_per_pixel, 0]

def draw_map(ax):
    ax.imshow(image, extent=EXTENT)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
```

Same as in step 1. `draw_map(ax)` draws the background into a plot so we don't repeat these lines every time.

---

### Cell 5 · Picture: training data and one example

```python
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

draw_map(axes[0])
for path in train["fut_map"][::5]:          # every 5th training future
    axes[0].plot(path[:, 0], path[:, 1], color="orange", linewidth=0.4, alpha=0.3)
axes[0].set_title("training futures")

i = 165                                      # one test example
axes[1].plot(test["hist_veh"][i, :, 0], test["hist_veh"][i, :, 1], "o-", color="tab:blue", markersize=3, label="history (input)")
axes[1].plot(test["fut_veh"][i, :, 0], test["fut_veh"][i, :, 1], "o-", color="tab:orange", markersize=3, label="future (sample)")
axes[1].plot(0, 0, "ks")
axes[1].set_aspect("equal")
axes[1].legend()
axes[1].set_title("one example in the vehicle frame")
plt.show()
```

`[::5]` = every 5th element. `alpha` = transparency, so thousands of lines don't hide the map.

---

**For the report:** nothing new here. This section only prepares the rest.
