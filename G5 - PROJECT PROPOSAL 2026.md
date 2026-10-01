**DM2583 – PROJECT PROPOSAL (revised 1 October 2026)**  
Group members: Fuad Mammadov, Gala Flores García, Harshit Poddar  
Contact person: Fuad Mammadov (fuadm@kth.se)

**PRELIMINARY TITLE**  
Constrained Generation of Vehicle Trajectories at an Urban Intersection using Flow Matching

**WHAT CHANGED IN THIS REVISION**

* All four strategies now use the same trained Flow Matching model. Strategy B is a soft constraint applied during sampling, not a penalty in the training loss.
* Exact dataset, recordings and number of usable samples are stated (measured from the files).
* One generated sample is defined exactly (60-dimensional, vehicle frame).
* Constraints are written mathematically as g(x) ≤ 0. The yaw-rate constraint is removed; the acceleration bound covers sharp turns.
* The drivable area is built from observed vehicle positions, with the Lanelet2 map as an optional check.
* The Frenet lateral-offset metric is replaced by a distance-to-road-edge metric that uses the same distance map as the constraint.

**BACKGROUND / ASSUMPTIONS**  
Motion planners and traffic simulators need a set of plausible futures for a vehicle, not only the most likely one. At an intersection a car may turn left, go straight or stop and wait. A model that regresses a single path tends to average these options. The result can be a path no driver would take.

Generative models can represent several manoeuvres at once. But nothing in sampling guarantees that a sample is physically possible: it can leave the road, exceed plausible speeds or change speed and direction too abruptly. We study how the point at which constraints are enforced affects the realism of the samples.

**DATA**

* Dataset: inD (RWTH Aachen / fka), https://www.ind-dataset.com. Drone recordings of four German intersections at 25 Hz, with position, velocity and acceleration of every road user, a Lanelet2 map and an aerial image per site. Access granted.
* Site: location 2 (Frankenburg, Aachen), recordings 18 to 29, about 4 hours. It has the most usable vehicles of the four sites and one map.
* We use one intersection only. The constraints depend on the road layout, so one fixed map keeps the drivable area and the projection operator simple.
* Filtering: classes car and truck_bus; vehicles that move less than 2 m are dropped as parked; tracks shorter than 9 s (history plus future) are dropped. Downsampled from 25 Hz to 5 Hz.
* Usable samples (counted from the files): 1,919 vehicles, about 14,500 windows with a 1 s stride.

| Split | Recordings | Vehicles | Windows |
|---|---|---|---|
| Train | 18 to 25 | 1,188 | about 8,900 |
| Validation | 26 to 27 | 365 | about 2,900 |
| Test | 28 to 29 | 366 | about 2,700 |

Windows from the same vehicle overlap, so the split is by recording, never by window. The independent unit is the vehicle.

**ONE GENERATED SAMPLE**

* Output: the next 6 s of one vehicle's centre position at 5 Hz, 30 points (x, y), so x ∈ R^60.
* Condition (input, not generated): the previous 3 s, 15 points (x, y) plus velocity.
* Coordinates: vehicle frame. The last observed position is the origin and the x axis points along the last observed heading. Samples are rotated back to map coordinates (metres) before constraints and metrics are evaluated.

**CONSTRAINTS**

Let p_0 be the last observed position and p_1 … p_30 the generated positions in map coordinates, Δt = 0.2 s. Let D(p) be the signed distance from p to the drivable region (negative inside, positive outside), read from a precomputed grid with 0.2 m cells.

1. Drivable area: g_road,k(x) = D(p_k) ≤ 0, for k = 1…30.
2. Speed: g_speed,k(x) = ‖p_k − p_(k−1)‖ / Δt − v_max ≤ 0, with v_max = 15 m/s.
3. Acceleration: g_acc,k(x) = ‖p_(k+1) − 2p_k + p_(k−1)‖ / Δt² − a_max ≤ 0, with a_max = 6 m/s².

A sample is feasible when every g ≤ 0 at every timestep (tolerance 1e-3). The bounds come from the training data. The fastest real vehicle reaches 13.9 m/s. The 99.9th percentile of acceleration is 5.1 m/s²; the few values above 6 m/s² are tracking noise. Real test trajectories are checked first and their feasibility rate is reported as the reference.

Drivable region: the union of grid cells that training vehicles' centres pass through, dilated by 1.5 m and with small holes filled. This is aligned with the trajectories by construction. The Lanelet2 map is overlaid as a visual check.

Projection Π (used by C and D), all operations vectorised in PyTorch:

1. Road: move every point with D(p_k) > 0 to the nearest drivable cell (nearest-index lookup from the distance transform).
2. Kinematics: walking forward from p_0, shorten any step longer than v_max·Δt and limit any second difference larger than a_max·Δt².
3. Repeat steps 1 and 2 three times, because fixing one constraint can break another.

**THE QUESTION TO BE EXAMINED**  
With one trained Flow Matching model, how does the point at which constraints enter sampling affect the feasibility and realism of the generated trajectories?

1. How much does each strategy reduce violations of the road, speed and acceleration constraints?
2. What does each strategy cost in fidelity, measured against held-out real trajectories?
3. Do the patterns seen on the torus (an equality constraint, a surface) appear for trajectories (inequality constraints, a region)?

**METHOD**

Flow Matching. Training pairs noise x_0 ~ N(0, I) with a real future x_1 and uses the linear path x_t = (1 − t)·x_0 + t·x_1. A network v_θ(x_t, t, history) learns to predict x_1 − x_0. Sampling integrates dx/dt = v_θ from t = 0 to 1 with 50 Euler steps. At any step the clean estimate is x̂_1 = x_t + (1 − t)·v_θ and the noise estimate is x̂_0 = x_t − t·v_θ.

Model: an MLP with a time embedding and a small history encoder, about one million parameters. It is trained once per seed with the plain Flow Matching loss. All four strategies reuse this model.

Four strategies (sampling only):

* A. Unconstrained: plain Euler integration.
* B. Soft: at each step the velocity is corrected by the gradient of a penalty, v ← v_θ − λ·∇_(x_t) P(x̂_1), with P(x) = Σ_k Σ_c max(0, g_c,k(x))². The road term is differentiable through bilinear interpolation of the distance grid. We test λ at four values chosen on the validation set.
* C. Repeated hard: at each step, x̂_1 ← Π(x̂_1) and the state is rebuilt as x_t = (1 − t)·x̂_0 + t·x̂_1 before the next step (the projected diffusion idea of Christopher, Baek and Fioretto, NeurIPS 2024, carried over to the Flow Matching ODE).
* D. Terminal: plain integration, then x_1 ← Π(x_1) once.

Problem 1 (torus): the same four strategies on the provided torus problem, with the same code structure (penalty for B, projection onto the surface for C and D).

**EVALUATION**  
Every strategy uses the same model, seeds (0, 1, 2), test histories and sample counts. Results are reported as mean ± standard deviation over three seeds.

Feasibility:

* Mean Constraint Violation: the average of max(0, g) over samples and timesteps, reported separately per constraint in its own unit (m, m/s, m/s²).
* On-Surface Rate: the fraction of samples with every constraint satisfied at every timestep.

Fidelity:

* Paired excess Sliced Wasserstein Distance: SWD between one generated sample per test history and the real futures (60-dim, map frame), minus the SWD between two halves of the real test set at the same size.
* Road-edge distance (analogue of the torus poloidal metric): the 1-Wasserstein distance between generated and real distributions of −D(p), the distance of points from the road edge. This shows whether a strategy piles samples up at the boundary or squeezes them to the middle.
* Diversity: the spread of endpoints over 20 samples per history, for 200 test histories.
* Manoeuvre shares: left, straight, right or stopped, from the heading change between the start and end of the path, compared with real data.
* Jerk distribution: to detect kinks created by projection.

Optional, if time allows: lateral offset from Lanelet2 lane centrelines.

**HYPOTHESIS**  
A gives the most varied manoeuvres and the most violations. C gives the best feasibility. Because nearest-point projection places corrected points on the road edge, we expect C and D to show an excess of points close to the boundary in the road-edge metric, with D worse, since a single late correction cannot be smoothed by later steps. D should also show jerk spikes where corrections land. B should trade feasibility for fidelity smoothly as λ grows. If C preserves both diversity and the road-edge distribution, that is a stronger result than the one we expect.

**SCOPE AND RESOURCES**  
Training is 3 runs (one model per seed), each a few minutes on a laptop or a free Kaggle GPU. All other comparisons are sampling runs on the same models. Out of scope: other road users, traffic lights, images, multiple intersections.

**TIMELINE**

* By 2 October (Stage 1 checkpoint): torus with strategies A to D and the four torus metrics.
* 2 to 5 October: data pipeline (windows, splits, drivable grid), bounds and real-data feasibility check, trajectory model trained on one seed.
* 6 to 9 October: strategies B, C, D on trajectories, metrics, full run over 3 seeds, results table.
* 10 to 12 October: figures and demo; code freeze 12 October.
* 13 October: Stage 2 checkpoint. 14 October: code and demo due. 15 October: report and presentation.

Demo: the same observed histories with generated paths from A, B, C and D side by side on the aerial image, with violating points in red.

**THE FINAL PAPER**  
A technical report of about 15 pages, a 20-minute presentation and a live demo.

**STUDENT BACKGROUND AND ROLES**

* Fuad Mammadov: MSc Interactive Media Technology, B.Eng Mechatronics and Robotics. Metrics, experiment runner, figures, demo.
* Gala Flores García: MSc Interactive Media Technology, Bachelor in Computer Science and Technology Engineering. Data pipeline: parsing inD, windows, splits, drivable grid.
* Harshit Poddar: MSc Machine Learning, B.Tech Electronics and Communication Engineering. Torus benchmark, Flow Matching model, shared penalty and projection module.

Everything lives in one shared repository with experiment settings under version control, so every number in the report can be reproduced with one command.
