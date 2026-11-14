# BRINK

Code for *Bounding operating limits from both sides by simulating infeasible designs* (Vineet Jagadeesan Nair, Stephen Peng, Arun Ravindran, submitted to L4DC 2027).

BRINK (Bounds from Rejected INfeasible designs with Kernel regression) estimates operating limits using simulations of both feasible and infeasible designs.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 and package versions are pinned in `.python-version` and `uv.lock`. Run commands from the repository root.

```bash
uv sync --locked
uv run pytest
```

For the Hopper experiments, download the pretrained [SAC policy](https://huggingface.co/sb3/sac-Hopper-v3):

```bash
uv run python scripts/hopper_setup.py download-policy
```

The download checks the pinned file's SHA-256. The upstream model declares no license and is not included in this repository.

## Quick example

Run two double-integrator baselines at three seeds (under a minute):

```bash
uv run python scripts/double_integrator_sweep.py sweep --workers 4 \
    --labels uncertified-sup-bisection uncertified-direct-score --seeds 400 401 402
```

## Reproduce the paper

Configurations and seeds are in [`configs/`](configs/). Runs write outputs and provenance to `results/`. [`configs/runs.yaml`](configs/runs.yaml) selects the latest run for each stage by default. Set explicit paths there to use existing runs instead.

Run the blocks below in order. The full reproduction takes about **1,100 core-hours**. Set `N` to your number of parallel workers.

### Main experiments

```bash
N=4

uv run python scripts/double_integrator_sweep.py sweep --workers "$N"
uv run python scripts/double_integrator_sweep.py analyze

for study in transmission feeder hopper covasim; do
    uv run python "scripts/${study}_setup.py" truth --workers "$N"
    if [ "$study" = hopper ]; then
        uv run python scripts/hopper_setup.py desk-check --workers "$N"
    else
        uv run python "scripts/${study}_setup.py" monotone-check --workers "$N"
    fi
    uv run python "scripts/${study}_setup.py" draw-columns
    uv run python "scripts/${study}_setup.py" score-columns --workers "$N"
    uv run python "scripts/${study}_sweep.py" sweep --workers "$N"
    uv run python "scripts/${study}_sweep.py" analyze
done
```

The studies cover a double integrator, transmission and distribution networks (`feeder`), MuJoCo Hopper, and Covasim. Sweeps use the supplied calibrations in `configs/frozen/`. Each script's `--help` lists calibration commands and other options. Sweep filters `--labels`, `--budgets`, and `--seeds` select smaller runs.

### Additional experiments

```bash
# Double-integrator replay with certified constants (Appendix B)
for stage in paths constants select replay; do
    uv run python scripts/double_integrator_certified.py "$stage" --workers "$N"
done
uv run python scripts/double_integrator_certified.py analyze

# Independent checks of recommended designs
for study in feeder hopper; do
    uv run python scripts/check_layer_sweep.py sweep --study "$study" --workers "$N"
    uv run python scripts/check_layer_sweep.py analyze --study "$study"
done

# GP-UCB and differential-evolution baselines
uv run python scripts/adaptive_baselines_sweep.py sweep \
    --studies transmission feeder hopper covasim --workers "$N"
uv run python scripts/adaptive_baselines_sweep.py analyze

uv run python scripts/hopper_setup.py witness-diagnostic --workers "$N"
```

### Figures and tables

After the experiments, generate the exhibits and summary statistics:

```bash
uv run python scripts/paper_figures.py
uv run python scripts/decisions_main.py
uv run python scripts/evaluation_dimensions.py
uv run python scripts/method_comparison.py
```

Figures and tables go to `exhibits/figures/` and `exhibits/tables/`. These scripts read stored runs only.

### Reproducibility

- Reported feeder experiments used arm64. Other studies used x86-64. Hopper rollouts and some reference-limit searches differ across architectures.
- Time-limited infeasibility tests can vary with host load, even with fixed seeds.
- Applied experiments use fitted bounds. Their observed coverage is empirical, not a mathematical guarantee.

## Citation

```bibtex
@unpublished{nair2027bounding,
  title  = {Bounding operating limits from both sides by simulating infeasible designs},
  author = {Jagadeesan Nair, Vineet and Peng, Stephen and Ravindran, Arun},
  note   = {Submitted to the Conference on Learning for Dynamics and Control (L4DC)},
  year   = {2027}
}
```

## License

[MIT](LICENSE).
