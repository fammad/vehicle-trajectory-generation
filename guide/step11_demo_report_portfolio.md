# Step 11 · Demo, report, portfolio

**Where:** Colab (demo cells) + writing · **Time:** demo 1 hour, report the rest

Everything the project produced ends up in three places: the **live demo in class**, the **report** (max 15 pages) and the **portfolio** (GitHub + website).

---

## Part A · Demo cells (end of the notebook)

Markdown cell: `# Part 3 · Demo`

### Cell 1 · Interactive demo: pick a car with a slider

```python
from ipywidgets import interact, IntSlider, FloatSlider

def live_demo(car=165, lam=1.0):
    demo_figure(car, lam=lam)

interact(live_demo,
         car=IntSlider(min=0, max=len(test["fut_veh"]) - 1, step=1, value=165, continuous_update=False),
         lam=FloatSlider(min=0.1, max=10, step=0.1, value=1.0, continuous_update=False))
```

- `interact` makes sliders that call `live_demo` again whenever you let go of a slider.
- `continuous_update=False`: only redraw when you release the slider (sampling C takes a second).
- In class: move the car slider to a few cars you picked before (write their numbers down), and drag λ to show B getting stricter.

### Cell 2 · Animation: from noise to paths

```python
from matplotlib.animation import FuncAnimation, PillowWriter

def make_animation(i, strategy="A", n=20, filename=None):
    c, origin, yaw, p0 = tensors_for(i, n)
    paths, frames = sample(model, c, origin, yaw, p0, strategy=strategy, keep_frames=True, seed=1)
    centre = test["hist_map"][i, -1]

    fig, ax = plt.subplots(figsize=(6, 6))
    def draw(k):
        ax.clear()
        draw_map(ax)
        ax.plot(test["hist_map"][i, :, 0], test["hist_map"][i, :, 1], "o-", color="tab:blue", markersize=3)
        for path in frames[k]:
            ax.plot(path[:, 0], path[:, 1], color="orange", linewidth=1)
        ax.set_xlim(centre[0] - 30, centre[0] + 30)
        ax.set_ylim(centre[1] - 30, centre[1] + 30)
        ax.set_title(f"strategy {strategy}, step {k + 1} of {len(frames)}")
    animation = FuncAnimation(fig, draw, frames=len(frames))
    animation.save(filename or OUTPUTS + f"flow_{strategy}_{i}.gif", writer=PillowWriter(fps=10))
    plt.close(fig)

make_animation(165, "A")
make_animation(165, "C")
```

- `FuncAnimation` calls `draw(k)` for every frame and glues the pictures into a GIF.
- The GIF shows Flow Matching at work: 20 random scribbles slowly turn into 20 possible futures. With C you also see the paths being kept on the road.
- Open the GIF from Drive (`outputs/`). Use it in the slides and on GitHub.

---

## Part B · Class demo (about 3 minutes of the 20-minute talk)

Prepare the day before:
1. Run the whole notebook once, so all models are saved in `outputs/`. On demo day it only loads (fast).
2. Write down 3 car numbers that show the differences well (one turn, one straight, one where A goes off road).
3. Keep the GIF and the demo figures in the slides as a backup if Wi-Fi or Colab fails.

Demo script:
1. *"This is one car, 3 seconds of history in blue."* Show the slider at car 1.
2. *"The model draws 20 possible futures. A has no rules: red paths leave the road."*
3. *"C repairs at every step: all orange, but look how they bunch together."* (diversity)
4. *"D repairs once at the end: on the road, but look at the kinks."*
5. Drag λ for B: *"soft push: the stronger, the fewer red paths."*
6. Show the GIF: *"this is what sampling looks like, from noise to path."*

---

## Part C · Report (max 15 pages)

Write it as you go: every step guide ends with "For the report". Suggested pages:

| Section | Pages | Content | From |
|---|---|---|---|
| 1 Introduction | 1 | why plausible futures matter, why generated paths can break physics, our question | proposal |
| 2 Background | 1.5 | Flow Matching, constrained generation, Christopher et al. 2024, trajectory prediction | proposal |
| 3 Data | 2 | inD, Frankenburg, cleaning steps, windows, vehicle frame, split table | steps 1-2 |
| 4 Method | 3 | the sample, the model, the 3 rules (table + formulas), penalty, projection, the 4 strategies, clean guess | steps 3, 5, 6, 7 |
| 5 Toy problem: torus | 1 | setup, results table, 2 figures, equality vs inequality | step 10 |
| 6 Evaluation | 1 | metric table, real-data reference row | step 8 |
| 7 Results | 3 | results table, trade-off figure, diversity/jerk figure, 2 demo figures, what we see per strategy | step 9 |
| 8 Discussion and limitations | 1.5 | hypothesis vs results, road grid from observed cars, one intersection, no other road users, projection is approximate | all |
| 9 Conclusion | 0.5 | answer the question in 4-5 sentences | |
| References + appendix | 0.5 | | |

Figures to use (all in `figures/` or `outputs/`): map with tracks, windows, vehicle frame, distance map, limits, first samples, projection, demo (2), trade-off, diversity/jerk, torus samples, torus poloidal. That is 13; drop the weakest if space is tight.

Words to always use the same way: sample, condition, window, vehicle frame, rule/constraint, feasible, projection Π, clean guess `x̂1`, on-surface rate, excess SWD.

---

## Part D · Portfolio

**GitHub (the repo you already have):**
1. Keep only clean notebooks: `offline/01-03`, `colab/DM2583_project.ipynb` (download from Colab with outputs).
2. Put the best pictures in `docs/`: one demo figure, the trade-off plot, one GIF.
3. Rewrite `README.md` as a project page:
   - one sentence: what it does
   - the GIF at the top
   - the problem (3 lines), the method (the A/B/C/D table), the result (trade-off figure + 3 bullet findings)
   - how to run (setup + data link: inD needs a request form, so do not upload the data)
   - team names and course
4. Add the final report as a PDF in `docs/`.

**Website (fammad.com):** a short case study page: problem → approach → one demo GIF → result in one picture → what I did myself → link to GitHub and the PDF.

---

**Done when**
- [ ] slider demo works on a fresh Colab session (only loading, no training)
- [ ] 2 GIFs and 2-3 demo figures saved
- [ ] report draft has every section from the table
- [ ] README rewritten with GIF and results
