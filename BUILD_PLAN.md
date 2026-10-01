# Build plan

Group 5 · DM2583 · Constrained Flow Matching for vehicle trajectories
Deadlines: torus + real application checkpoint **13 Oct** · code + demo **14 Oct** · report + talk **15 Oct**

---

## The project in 5 sentences

1. Drone videos of one intersection (inD, Frankenburg) give the position of every car 25 times per second.
2. We cut every car's track into pieces: **3 s of history** (input) and **6 s of future** (what we generate).
3. A **Flow Matching** model learns to draw a possible future from the history.
4. Some drawn futures are impossible: off the road, too fast, too jerky. We write these as **3 rules**.
5. We compare **4 ways (A, B, C, D)** to make the model follow the rules and measure what each costs. The same on a **torus** toy problem.

---

## How the guides work

Every step has its own guide in `guide/`. Each guide gives you the notebook cells **one by one**:
the code, then what every line does, then what you should see. Every step ends with a **picture**, a **check**, and a **"for the report"** note.
Type the cells yourself (or copy them), run them, look at the result, then go on. If something breaks, compare with the guide.

All code was run once from start to end before writing the guides, and the numbers you should see come from that run.

**Rules for the code:** only `numpy`, `pandas`, `matplotlib`, a few `scipy` functions and `torch`. Short functions with comments. One class only for each model. If simple is good enough, simple wins.

---

## The steps

| Step | Guide | Where | What you get | Time |
|---|---|---|---|---|
| 0 | this file (below) | laptop | Python + Jupyter ready | 20 min |
| 1 | `guide/step1_explore.md` | laptop | understand the files, map with all cars | 1 h |
| 2 | `guide/step2_samples.md` | laptop | `train/val/test.npz` (14,540 windows) | 2 h |
| 3 | `guide/step3_road_and_rules.md` | laptop | road grid, 3 rules, 99% check, `road_grid.npz` | 1.5 h |
| – | upload `data_processed/` to Drive | | | 5 min |
| 4 | `guide/step4_colab_load.md` | Colab | notebook connected to Drive, data loaded | 30 min |
| 5 | `guide/step5_model.md` | Colab | trained Flow Matching model, first generated paths | 2 h |
| 6 | `guide/step6_rules_torch.md` | Colab | rules with gradients, projection Π | 1.5 h |
| 7 | `guide/step7_strategies.md` | Colab | `sample()` with A/B/C/D, the demo figure | 1.5 h |
| 8 | `guide/step8_metrics.md` | Colab | all metrics + real-data reference | 1.5 h |
| 9 | `guide/step9_experiment.md` | Colab | 3 seeds, results table, trade-off + diversity figures | 1 h + GPU |
| 10 | `guide/step10_torus.md` | Colab | torus with A/B/C/D, 4 torus metrics, 2 figures | 2 h |
| 11 | `guide/step11_demo_report_portfolio.md` | Colab + writing | slider demo, GIF animation, report outline, portfolio | rest |

Steps 1-3 need the raw data (1.2 GB) so they run on the laptop. Steps 4-11 live in **one Colab notebook** that the whole group shares.
Step 10 (torus) only needs steps 4-5 done; someone can do it in parallel with steps 6-9.

---

## Step 0 · Setup (once)

```bash
cd ~/Desktop/"DM2583 - Big Data"/Project-vehicle-trajectory-generation
python3.11 -m venv .venv           # a separate box for this project's libraries (brew install python@3.11 if missing)
source .venv/bin/activate          # step into the box. Do this every time you open a new terminal
pip install -r requirements.txt    # numpy, pandas, matplotlib, scipy, jupyter
jupyter notebook                   # opens Jupyter in the browser
```

In Jupyter: open `offline/`, New → Python 3. **Check:** a cell with `import numpy, pandas, matplotlib, scipy` runs without errors.

---

## Folder layout at the end

```
Project-vehicle-trajectory-generation/        (GitHub repo)
├── BUILD_PLAN.md               this file
├── guide/                      step-by-step guides
├── offline/                    01_explore_data, 02_make_samples, 03_road_and_rules (.ipynb)
├── colab/DM2583_project.ipynb  downloaded copy of the shared Colab notebook
├── figures/                    pictures from steps 1-3
├── docs/                       best figures, GIF, final report PDF (for the portfolio)
├── data/, data_processed/      not in git
└── README.md                   becomes the portfolio page at the end

Google Drive: Group 5 - Big Data/DM2583_project/
├── DM2583_project.ipynb        the shared notebook
├── data_processed/             from step 3
├── raw/                        18_background.png, 18_recordingMeta.csv
└── outputs/                    models, results.csv, all figures and GIFs from Colab
```

---

## What the end looks like

- **Class demo:** sliders to pick a car and change λ, 4 panels A/B/C/D live, plus a GIF of paths forming from noise. (step 11)
- **Report (15 pages):** page plan and figure list in step 11. Every guide ends with what goes into which section.
- **Portfolio:** README with GIF and results, report PDF on GitHub, short case study on fammad.com. (step 11)

---

## Git habit

After every step: `git add <files> && git commit -m "step N: ..." && git push`.
Never `git add data/` or `data_processed/` (the `.gitignore` already blocks them).
