#!/bin/bash
# slurm_job.sh
# Cal Poly HPC SLURM job script for the full benchmark sweep.
# Submit with: sbatch scripts/slurm_job.sh
#
# Before submitting:
#   1. Replace <YOUR_ACCOUNT> with your HPC account string
#   2. Replace <YOUR_EMAIL> with your email
#   3. Confirm partition name: `sinfo` on the login node
#   4. Confirm MPI module name: `module avail` on the login node (look for openmpi/mpich)
#   5. Run the pre-flight check manually first (see below)

#SBATCH --job-name=mm_perf_study
#SBATCH --account=<YOUR_ACCOUNT>       # TODO: fill in
#SBATCH --partition=compute            # TODO: confirm with `sinfo`
#SBATCH --nodes=1                      # Single node — keeps MPI latency low
#SBATCH --ntasks=16                    # Max MPI processes needed (P_max=16 per assignment)
#SBATCH --cpus-per-task=1              # One CPU per MPI rank
#SBATCH --mem=16G                      # 3× 2048×2048 double matrices ≈ 96MB; 16G is very safe
#SBATCH --time=02:30:00                # Estimated: ~2hr for full sweep with REPEATS=5
#SBATCH --output=results/slurm_%j.out
#SBATCH --error=results/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>       # TODO: fill in

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
module purge
module load gcc/12.2.0    # TODO: confirm exact version with `module avail gcc`
module load openmpi       # TODO: confirm exact name with `module avail` (may be openmpi/4.x or mpich)

echo "========================================"
echo " Job ID  : ${SLURM_JOB_ID}"
echo " Node    : $(hostname)"
echo " Tasks   : ${SLURM_NTASKS}"
echo " Start   : $(date)"
echo "========================================"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p results/raw results/plots

# ============================================================
# BUILD ON COMPUTE NODE
# (Ensures correct CPU architecture flags for this node)
# ============================================================
echo ""
echo "[1/3] Building..."
make clean && make
echo "Build complete."

# ============================================================
# PRE-FLIGHT CHECK
# Run one config to verify timing output before the full sweep.
# If TIME: is missing or the binary crashes, abort here.
# ============================================================
echo ""
echo "[2/3] Pre-flight: testing 2d 512x512x512 P=4..."
PREFLIGHT_OUT=$(mpirun -np 4 ./mm 2d 512 512 512 4 2>&1)
echo "$PREFLIGHT_OUT" | tail -5

if ! echo "$PREFLIGHT_OUT" | grep -q "TIME:"; then
    echo ""
    echo "ERROR: Pre-flight FAILED — no 'TIME:' in output."
    echo "       Aborting job to avoid wasting allocation."
    exit 1
fi
echo "Pre-flight OK — timing output found."

# ============================================================
# FULL BENCHMARK SWEEP
# ============================================================
echo ""
echo "[3/3] Running full sweep (REPEATS=5, all sizes including 2048)..."

export BINARY="./mm"
export REPEATS=5
export WARMUP=2

bash scripts/run_experiments.sh

echo ""
echo "========================================"
echo " Job complete: $(date)"
echo " Results in  : results/raw/"
echo " Next step   : python3 scripts/analyze.py results/raw/*.csv"
echo "========================================"
