# AI-Driven Task Offloading System

A machine learning project that teaches a **Random Forest Classifier** to decide where a computational task should run — locally on the device, or offloaded to one of two edge servers or the cloud — based on real-time device and network conditions.

**Course:** Artificial Intelligence & Machine Learning  
**Team:** Taha Muzaffar · Muhammad Umer Farooqui · Maham Faisal

---

## What This Project Does

When a device needs to run a task, it has options: run it locally (fast but drains the CPU and battery), or ship it off to a more powerful machine somewhere on the network. Making that call intelligently — not just randomly or always the same way — is the problem this project solves.

The system looks at six things about a task and the device at that moment, and outputs one of four destinations:

| Target | Code | When it's chosen |
|--------|------|-----------------|
| Local Device | 0 | CPU and battery are healthy, network latency is low |
| Edge Server 1 | 1 | CPU is high but task is small/simple |
| Edge Server 2 | 2 | CPU is very high or task is larger |
| Cloud | 3 | Everything else — last resort |

---

## How It Works

### 1. Data Simulation

There's no live hardware here, so the project generates synthetic telemetry data mimicking a real multi-user environment. For each task, six features are randomly sampled within realistic ranges:

- **CPU Usage (%)** — how loaded the device processor is
- **Battery Level (%)** — remaining charge
- **Network Latency (ms)** — round-trip delay to the server
- **Bandwidth (kbps)** — available throughput
- **Data Size (KB)** — how much data the task involves
- **Task Complexity** — Low / Medium / High

A heuristic rule-set (if-else logic with thresholds) then labels each task with the correct offload target. This labeled dataset becomes the training data.

### 2. Training the Model

A **Random Forest Classifier** (100 decision trees) is trained on 80% of the simulated data. Task complexity is encoded numerically before training (`low=0, medium=1, high=2`). The remaining 20% is held out for evaluation.

Random Forest was chosen because it handles mixed feature types well, is robust to noisy simulation data, and naturally provides feature importance scores.

### 3. Latency Analysis

Prediction alone isn't enough — the project also estimates the real cost of each offloading decision by computing **Total Response Time**:

```
Total Time = Upload Time + Processing Time + Download Time
```

Processing time varies by destination (edge servers are faster than cloud), and upload/download speeds are randomized to reflect variable network conditions. Local execution has zero transfer time but its own processing cost.

### 4. Evaluation

The model is evaluated with a classification report and a confusion matrix, showing how well it distinguishes between the four destinations. A feature importance bar chart also reveals which inputs drive the decision most — typically CPU usage and data size are the strongest signals.

---

## Project Structure

```
AIproject.ipynb
│
├── Library Imports
├── Data Simulation          # Generates synthetic telemetry + heuristic labels
├── Visualize the Data       # Pairplot colored by offload target
├── Prepare Data             # Encodes features, splits 80/20
├── Train the Model          # Random Forest (100 estimators)
├── Latency Analysis         # Simulates upload/processing/download times
├── Evaluate the Model       # Confusion matrix + feature importance
├── Visualize Latency        # Stacked bar charts per user
└── Test Run                 # 5 custom tasks → live predictions + latency
```

---

## Test Run

The final section lets you define your own tasks manually and watch the model decide where to run them. Five custom scenarios are included that cover edge cases:

| User | Condition | Expected Behaviour |
|------|-----------|--------------------|
| 1 | CPU at 85%, large task | Offload to Server 2 |
| 2 | Battery at 15% | Offload (low battery) |
| 3 | Latency at 270ms | Offload (poor network) |
| 4 | CPU 20%, battery 90%, low latency | Run locally |
| 5 | Moderate conditions | Likely local or Server 1 |

After prediction, upload/processing/download latency is computed and visualized only for the tasks that were offloaded.

---

## How to Run

1. Open `AIproject.ipynb` in Google Colab or Jupyter.
2. Run all cells top to bottom — each section builds on the previous.
3. To test your own tasks, edit the `custom_tasks` DataFrame in the **Test Run** section with your own values and re-run those cells.

### Dependencies

All standard — no special installs needed in Colab:

```
numpy, pandas, matplotlib, seaborn, scikit-learn
```

---

## Limitations

- Data is entirely simulated; the model's real-world accuracy depends on how well the simulation reflects actual device behavior.
- The heuristic labeling means the ML model is essentially learning a rule-set — the value would be greater with real measured data.
- Latency simulation uses random speeds, so results vary between runs.
