# Individual Project: CartPole + Acrobot

## Tools
64-bit Python 3.11, PowerShell terminal, and optionally VS Code and Git.

You do not need CUDA, Colab, service accounts, or an API key. CPU is the default.

Extract the ZIP file and open a terminal inside this folder (where `run.py` is located).

## Installation on Windows
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux: `python3 -m venv .venv`; use `.venv/bin/python` instead of the Windows executable.

The dependency version ranges allow installation; each run saves a `requirements_lock.txt` file containing the exact versions. To reproduce the environment, use that file in a clean environment.

## Quick Verification (Not Scientific Evidence)
```powershell
.\.venv\Scripts\python.exe run.py --smoke
.\.venv\Scripts\python.exe plot.py --root results_smoke
.\.venv\Scripts\python.exe video.py results_smoke/cart_dqn/seed_11
```

Runs all ten configurations with three seeds and only 1,200 steps. Even if little learning occurs, this verifies parameter updates, file writing, evaluation, and figure generation. It may take several minutes due to evaluations.

## Full Experiments
```powershell
.\.venv\Scripts\python.exe run.py --suite tier1
.\.venv\Scripts\python.exe run.py --suite tier2
.\.venv\Scripts\python.exe plot.py
.\.venv\Scripts\python.exe video.py results/cart_dqn/seed_11
.\.venv\Scripts\python.exe video.py results/ac_base/seed_11
```

You can also run both tiers using `run.py --suite all`.

For a single configuration: `run.py --experiment ac_base` (three seeds).

For an initial measurement: `run.py --experiment ac_base --seeds 11`.

If you subsequently run tier2, any existing completed run will be skipped.

To reproduce each figure: tier1 requires `--suite tier1`; the other three figures require `--suite tier2`; then run `plot.py`.

To reproduce everything from scratch, use `--out results_repeat` and `plot.py --root results_repeat`.

## Experimental Design

- Seeds 11, 22, and 33 are used for Python, NumPy, PyTorch, and the environment.
- CartPole: 200,000 steps per run for both algorithms; 6, 10, and 20 bins per dimension. Ranges: position ±2.4, velocity ±3, angle ±0.2095, and angular velocity ±3.5. Values outside these ranges are assigned to the extreme bins; this saturation is a limitation that should be discussed.
- Acrobot: 300,000 steps per run; this is an initial training budget, not a guarantee of success.
- DQN: two hidden layers of 64 units, ReLU, Adam, Huber loss, replay buffer of 50,000 transitions, minibatch size of 64, gamma of 0.99, and one update every four steps after an initial 1,000-step warm-up.
- The target network is copied every 1,000 steps. Ablation: the online network is used to compute the target, without gradients; everything else remains unchanged.
- Linear epsilon decay from 1 to 0.05 over half of the training budget, followed by a constant value. This is not presented as GLIE.
- Learning rate study: 0.0003, 0.001, and 0.003. Decay fraction study: 0.25, 0.5, and 0.75. One variable is changed at a time, using a shared baseline.
- Evaluation every 10,000 steps using 10 greedy episodes; a separate environment and fixed evaluation seeds are used for comparability. Final evaluation uses 100 episodes with different seeds.
- Curves: mean ± one standard deviation ACROSS seeds; this is not a confidence interval.
- `first_threshold_10ep` represents only the first provisional threshold crossing based on 10 episodes. The `success` column verifies the threshold using the final 100 episodes; it does not demonstrate sustained performance.
- Assignment thresholds: CartPole 475; Acrobot -100. Training and evaluation returns are stored separately.
- Restarting skips completed runs, but an interrupted run starts again from scratch. Final models are not checkpoints for resuming training.
- Modify budgets and parameters in `configs()` within `run.py` BEFORE launching experiments; use a different `--out` folder for a new experimental design. Do not mix results from different protocols.

## Generated Files

`results/<experiment>/seed_<n>/` contains `config.json`, `training.csv`, `evaluation.csv`, `summary.json`, and either `model.pt` or `q.npy`. The `video.py` script adds `agent.gif`.

`results/figures` contains four plots; `summary.csv` aggregates results, variability, timing, and thresholds.

`summary.json` preserves the 100 final episode returns, total/evaluation/training times, and seed.

`hardware.json` records system and device information; manually add the CPU model, RAM, and power mode if the system does not identify them.

Training time excludes evaluations; `wall_seconds` includes final evaluation and model saving. Record both when making comparisons.

## What to Expect

The program displays the experiment name, seed, steps, and mean return. It does not open windows during training.

CartPole may improve toward 500, while Acrobot may achieve less negative returns; curves may fluctuate, and some configurations may fail. No performance threshold is guaranteed.

The 30 training runs total 7.8 million training steps, plus evaluation. Measure one run first to estimate the actual execution time. Previously discussed timing estimates are approximate.

Do not select only the best seed for the tables. GIFs use a fixed illustrative seed and do not replace evaluations.

## Pending Academic Deliverables

This package provides experimental infrastructure, NOT a research paper with results already obtained.

After training, the remaining tasks include interpreting the results, writing the 6–8-page double-column PDF, preparing a 1–2-page theoretical analysis, including an AI disclosure statement and an appendix with prompts, providing the Git repository, and preparing for the defense.

Paper structure: title, abstract, introduction, relevance, methods, experimental setup, results, discussion, analysis of continuous actions/pixel observations, limitations, AI disclosure statement, and verified references.

Do not fabricate numbers or results. Save the prompts used, including the request that generated this package.

The defense requires explaining discretization, Bellman equations, epsilon, experience replay, target networks, termination vs. truncation, evaluation, and variability bands across seeds.

## References for Further Study

https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

https://gymnasium.farama.org/environments/classic_control/cart_pole/

https://gymnasium.farama.org/environments/classic_control/acrobot/

## CartPole Diagnostics and Revised Plots

Run the following commands from the project root with the virtual environment activated.

The first command reproduces the single-seed diagnostic using the reduced learning rate.

The second command reproduces the adjusted experiment using three seeds and target network updates every 10,000 steps.

```powershell
python -c "from pathlib import Path; from run import configs, run; c=configs()['cart_dqn']; c['lr']=0.0001; run('cart_dqn_lr0001', c, 11, Path('results_diagnostico'), False)"
python -c "from pathlib import Path; from run import configs, run; c=configs()['cart_dqn']; c['lr']=0.0001; c['target_every']=10000; [run('cart_dqn_lr0001_target10000', c, seed, Path('results_diagnostico'), False) for seed in [11,22,33]]"
python plot_revision.py
```

The revised plots and table are saved in `resultados_revisados/`.

The English report is located in `report/`, containing `main.tex` and the figures needed to compile it using pdfLaTeX.

The CartPole and Acrobot demonstrations are located in `evidencias/`.
