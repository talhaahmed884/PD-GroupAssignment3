# Group Assignment 3 — MPI Matrix Multiplication

CS 5350 Parallel and Distributed Algorithms  
Performance study comparing three MPI matrix multiplication implementations.

---

## Algorithms

| Name | Flag | Description |
|------|------|-------------|
| MM-ser | `ser` | Serial triple-loop baseline |
| MM-1D  | `1d`  | 1D row-strip decomposition with ring-shift (requires P divides m and n) |
| MM-2D  | `2d`  | 2D block SUMMA on a √P×√P process grid (requires P = perfect square) |

**Binary interface:**
```
mpirun -np <P> ./mm <algo> <m> <n> <q> <P>
```
Computes `A(m×n) × B(n×q) = C(m×q)` using `P` MPI processes.  
Always prints `TIME: <seconds>` to stdout on rank 0.

---

## Prerequisites

**macOS:**
```bash
brew install open-mpi python3
pip3 install pandas matplotlib scipy seaborn
```

**WSL / Ubuntu:**
```bash
sudo apt update && sudo apt install -y libopenmpi-dev openmpi-bin make python3 python3-pip
pip3 install pandas matplotlib scipy seaborn
```

**Windows Git Bash** (for the `run_wsl.sh` wrapper only):
- WSL installed with an Ubuntu distro

---

## Path note (Windows only)

The repo folder name contains parentheses which break shell expansion.  
Create a symlink in WSL once so the helper scripts can find the repo:

```bash
ln -s "/mnt/c/Users/<YOUR_WINDOWS_USERNAME>/path/to/GroupAssignment3" \
      "/mnt/c/path/to/GroupAssignment3-link"
```

Then update `REPO_WSL` at the top of `scripts/run_wsl.sh` to point to the symlink path.

---

## Build

**From Windows Git Bash:**
```bash
bash scripts/run_wsl.sh make
```

**From macOS / WSL / Ubuntu directly:**
```bash
make
```

This compiles all `.cpp` files in `src/` (excluding `correctness_check.cpp`) into `./mm` using `mpicxx`.

To also build the correctness checker:
```bash
make check-build   # produces ./correctness_check
```

To clean built binaries:
```bash
make clean
```

---

## Correctness checks

Run before any timing experiments. Validates `1d` and `2d` against the serial reference on small inputs.

**From Windows Git Bash:**
```bash
bash scripts/run_wsl.sh make check-build
bash scripts/run_wsl.sh run_correctness.sh
```

**From macOS / WSL / Ubuntu:**
```bash
make check-build
bash scripts/run_correctness.sh
```

Expected output ends with:
```
Results: 9 passed | 0 failed | 0 skipped
```

If any algorithm fails, fix it before running the benchmark sweep — speedup numbers from incorrect code are meaningless.

---

## Benchmark sweep

Sweeps over matrix sizes and process counts, recording timing to a CSV.

**From Windows Git Bash:**
```bash
bash scripts/run_wsl.sh run_experiments.sh
```

**From macOS / WSL / Ubuntu:**
```bash
bash scripts/run_experiments.sh
```

Output: `results/raw/results_YYYYMMDD_HHMMSS.csv`

### Configuration (edit top of `scripts/run_experiments.sh`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `REPEATS` | `3` | Timed runs per config (median used in analysis) |
| `WARMUP` | `1` | Warmup runs discarded before timing |
| `SIZES` | square + rectangular | Matrix dimension triples `"m n q"` |
| `P_VALUES` | `1 4 16` | Process counts (all perfect squares for MM-2D compatibility) |

The `2048×2048` size is commented out by default — uncomment only for HPC runs.

---

## Analyze results

```bash
python3 scripts/analyze.py results/raw/results_<timestamp>.csv
```

Produces plots and a summary table in `results/plots/`:

- `speedup_vs_P_size*.png` — speedup curves per matrix size
- `runtime_vs_size_P*.png` — runtime scaling per process count
- `cost_vs_P_size*.png` — cost (P × T) analysis
- `efficiency_heatmap_size*.png` — parallel efficiency heatmaps
- `shape_comparison.png` — rectangular vs. square shape sensitivity
- `summary_table.csv` — aggregated median times, speedups, costs

---

## Manual single run

```bash
# Serial, 512×512×512
mpirun -np 1 ./mm ser 512 512 512 1

# 1D row-strip, 4 processes
mpirun -np 4 ./mm 1d 512 512 512 4

# 2D SUMMA grid, 4 processes (P must be a perfect square)
mpirun -np 4 ./mm 2d 512 512 512 4

# 2D SUMMA grid, 16 processes
mpirun -np 16 ./mm 2d 512 512 512 16
```

Add `--oversubscribe` if your machine has fewer physical cores than `P`:
```bash
mpirun --oversubscribe -np 16 ./mm 2d 512 512 512 16
```

---

## Project structure

```
.
├── src/
│   ├── mm.cpp                  # All three algorithm implementations + main
│   └── correctness_check.cpp   # Validates output against serial reference
├── scripts/
│   ├── run_wsl.sh              # Windows Git Bash → WSL entry point
│   ├── run_experiments.sh      # Full benchmark sweep
│   ├── run_correctness.sh      # Correctness validation
│   ├── analyze.py              # CSV → plots + summary table
│   └── slurm_job.sh            # HPC SLURM job submission script
├── results/
│   ├── raw/                    # Timing CSVs (git-ignored except .gitkeep)
│   └── plots/                  # Generated graphs and summary table
└── Makefile
```

---

## HPC runs

Final reported results should be collected on Cal Poly's HPC center.  
Use `scripts/slurm_job.sh` as the job submission script.

Before submitting:
1. Fill in `<YOUR_ACCOUNT>` and `<YOUR_EMAIL>` in `slurm_job.sh`
2. Confirm the partition name with `sinfo` on the login node
3. Confirm the MPI module name with `module avail` on the login node
4. Uncomment the `2048×2048` size in `run_experiments.sh` for HPC runs only

```bash
sbatch scripts/slurm_job.sh
```
