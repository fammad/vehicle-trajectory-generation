# Step 1 · Explore the data

**Where:** laptop · **Notebook:** `offline/01_explore_data.ipynb` · **Time:** about 1 hour

**Goal:** understand what is inside the inD files before we change anything.
At the end you can say: "one row is one car in one moment, and here are all the cars drawn on the map".

Create the notebook: in Jupyter, go to the `offline/` folder → New → Python 3. Rename it `01_explore_data`.
Then add the cells below one by one. Run each cell (Shift + Enter) and look at the result before going on.

---

### Cell 1 · Imports (markdown title first)

Add a **markdown** cell at the top: `# Step 1: Explore the inD data`

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

- `numpy` (np): fast math on arrays of numbers.
- `pandas` (pd): reads tables (CSV files) into a `DataFrame`, like an Excel sheet in Python.
- `matplotlib.pyplot` (plt): draws plots.
- `as np` is only a short name, so we write `np.` instead of `numpy.`.

---

### Cell 2 · Load one recording

```python
DATA = "../data/"     # the raw files are one folder up, in data/
rec = 18              # recording number. 18 to 29 are all at the Frankenburg intersection

tracks = pd.read_csv(f"{DATA}{rec}_tracks.csv")
meta = pd.read_csv(f"{DATA}{rec}_tracksMeta.csv")
rec_meta = pd.read_csv(f"{DATA}{rec}_recordingMeta.csv")

print("tracks:", tracks.shape)
print("meta:", meta.shape)
```

- `f"{DATA}{rec}_tracks.csv"` is an *f-string*: Python puts the values in the `{}`, so it becomes `"../data/18_tracks.csv"`.
- Every recording has 3 CSV files:
  - `tracks`: one row = one road user in one frame (position, speed, ...). The big one.
  - `tracksMeta`: one row = one road user (its class: car, pedestrian, ...).
  - `recordingMeta`: one row = the recording itself (frame rate, duration, scale of the image).
- `.shape` gives (rows, columns).

**You should see** about 626,000 rows in `tracks` and 427 rows in `meta`.

---

### Cell 3 · What columns do we have?

```python
tracks.head()
```

`.head()` shows the first 5 rows. The columns we will use:

| Column | Meaning |
|---|---|
| `trackId` | which road user (same number = same car) |
| `frame` | which moment. 25 frames per second |
| `xCenter`, `yCenter` | position in metres |
| `heading` | direction the car points to, in degrees |
| `xVelocity`, `yVelocity` | speed in x and y, in m/s |

---

### Cell 4 · Who is in the recording?

```python
counts = meta["class"].value_counts()
print(counts)

counts.plot(kind="bar")
plt.title(f"Road users in recording {rec}")
plt.ylabel("count")
plt.show()
```

- `meta["class"]` takes one column.
- `.value_counts()` counts how often every value appears.
- `.plot(kind="bar")` is a pandas shortcut for a bar chart.

**You should see** cars, pedestrians and bicycles. We will only keep `car` and `truck_bus`.

---

### Cell 5 · Information about the recording

```python
print("frame rate:", rec_meta["frameRate"][0], "frames per second")
print("duration:", round(rec_meta["duration"][0] / 60, 1), "minutes")
print("metres per image pixel:", rec_meta["orthoPxToMeter"][0])
```

- `rec_meta["frameRate"][0]`: column `frameRate`, row 0 (there is only one row).

---

### Cell 6 · Load the aerial image and give it a size in metres

```python
image = plt.imread(f"{DATA}{rec}_background.png")

# The background image is 12 times smaller than the original drone image (this is how the inD tools do it),
# so one pixel of this image = orthoPxToMeter * 12 metres.
metres_per_pixel = rec_meta["orthoPxToMeter"][0] * 12
height, width = image.shape[:2]

# extent = [left, right, bottom, top] in metres.
# The y axis of inD points up, while image rows go down, so the image goes from 0 to minus its height.
extent = [0, width * metres_per_pixel, -height * metres_per_pixel, 0]

plt.figure(figsize=(9, 6))
plt.imshow(image, extent=extent)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Frankenburg intersection")
plt.show()
```

- `plt.imread` reads the PNG as an array of pixels.
- `image.shape[:2]` = (height, width) in pixels.
- `extent` tells matplotlib where the image lies in metres, so we can draw the car positions (also in metres) on top.

---

### Cell 7 · Draw every car on the map

```python
car_ids = meta[meta["class"] == "car"]["trackId"]

plt.figure(figsize=(9, 6))
plt.imshow(image, extent=extent)
for car_id in car_ids:
    car = tracks[tracks["trackId"] == car_id]
    plt.plot(car["xCenter"], car["yCenter"], color="orange", linewidth=0.7)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title(f"All {len(car_ids)} cars in recording {rec}")
plt.savefig("../figures/01_tracks_on_map.png", dpi=150, bbox_inches="tight")
plt.show()
```

- `meta[meta["class"] == "car"]` keeps only the rows where class is car. Then we take their `trackId`.
- In the loop, `tracks[tracks["trackId"] == car_id]` = all rows (all frames) of one car.
- `plt.savefig` saves the picture for the report. Do it **before** `plt.show()`.

**Check:** the orange lines must lie on the roads. If they do, the coordinates and the image match.

---

### Cell 8 · One car over time

```python
car = tracks[tracks["trackId"] == car_ids.iloc[0]]

time = car["frame"] / 25                                    # frame number -> seconds
speed = np.sqrt(car["xVelocity"]**2 + car["yVelocity"]**2)  # length of the velocity vector

plt.figure(figsize=(8, 3))
plt.plot(time, speed)
plt.xlabel("time (s)")
plt.ylabel("speed (m/s)")
plt.title("Speed of one car")
plt.show()

print("number of frames for this car:", len(car))
```

- `car_ids.iloc[0]` = the first car id in the list.
- Speed = √(vx² + vy²) (Pythagoras).
- 13.9 m/s = 50 km/h, the speed limit here.

---

### Done when

- [ ] you have the map with all car tracks (`figures/01_tracks_on_map.png`)
- [ ] you can explain: track, frame, recording, the three CSV files

**Commit:** `git add offline/01_explore_data.ipynb figures/ && git commit -m "step 1: explore data" && git push`

**For the report (Data section):** one sentence on inD, the map figure, the number of road users per class.
