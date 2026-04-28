#!/usr/bin/env bash
# run_experiments.sh
# Full benchmark sweep: varies matrix sizes and process counts, records timing to CSV.
# Run locally:  bash scripts/run_experiments.sh
# Run via WSL:  bash scripts/run_wsl.sh run_experiments.sh
#
# Requires: ./mm to be built (`make`) and mpirun to be available in PATH
# Produces: results/raw/results_YYYYMMDD_HHMMSS.csv

set -euo pipefail

# ============================================================
# CONFIGURATION — edit these to match teammate's interface
# ============================================================
BINARY="${BINARY:-./mm}"
REPEATS="${REPEATS:-3}"       # timed reps per config (median taken in analyze.py)
WARMUP="${WARMUP:-1}"         # warmup runs (discarded)

# Algorithm selector strings
ALGO_SER="${ALGO_SER:-ser}"
ALGO_1D="${ALGO_1D:-1d}"
ALGO_2D="${ALGO_2D:-2d}"

# Matrix sizes: "m n q" triples
# Square sizes for general scaling study; rectangular for shape sensitivity (Q2)
# NOTE: 2048 is HPC-only — comment it out for local runs to avoid multi-hour waits
SIZES=(
    "128 128 128"
    "256 256 256"
    "512 512 512"
    "1024 1024 1024"
    "2048 2048 2048"    # Uncomment for HPC runs only (serial alone ~60-240s)

    # Rectangular: thin-tall A — stresses 1D row decomposition
    "512 128 512"
    "1024 64 1024"

    # Rectangular: wide A — stresses column-heavy access patterns
    "128 512 128"
)

# P values: perfect squares required for MM-2D; P=1 is serial baseline only
# Power-of-2 matrix sizes divide evenly into 4 and 16 but not 9/25/36/49,
# so those are omitted to avoid expected ERROR rows in the CSV.
P_VALUES=(1 4 16)

# ============================================================
# OUTPUT SETUP
# ============================================================
OUTDIR="results/raw"
mkdir -p "$OUTDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="${OUTDIR}/results_${TIMESTAMP}.csv"

# ============================================================
# HELPERS
# ============================================================
is_perfect_square() {
    local n=$1
    local s
    s=$(python3 -c "import math; s=math.isqrt($n); print(1 if s*s==$n else 0)")
    [[ "$s" == "1" ]]
}

extract_time() {
    # Parses "TIME: 1.23456789" from program stdout.
    # Uses awk instead of grep -oP — macOS BSD grep does not support Perl regex (-P).
    echo "$1" | awk '/TIME:/ { print $2; exit }'
}

# ============================================================
# PREFLIGHT
# ============================================================
if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: Binary '$BINARY' not found."
    echo "       Run: bash scripts/run_wsl.sh make"
    exit 1
fi

echo "========================================"
echo " Benchmark Sweep"
echo " Binary  : $BINARY"
echo " Repeats : $REPEATS  Warmup: $WARMUP"
echo " Output  : $OUTFILE"
echo "========================================"

# CSV header
echo "algorithm,m,n,q,P,rep,time_seconds" > "$OUTFILE"

ALGO_LIST=("$ALGO_SER" "$ALGO_1D" "$ALGO_2D")
TIMING_WARN_SHOWN=0

# ============================================================
# MAIN SWEEP
# ============================================================
for SIZE in "${SIZES[@]}"; do
    read -r M N Q <<< "$SIZE"

    for P in "${P_VALUES[@]}"; do
        for ALGO in "${ALGO_LIST[@]}"; do

            # Serial: only run at P=1; skip all other P values
            if [[ "$ALGO" == "$ALGO_SER" && "$P" -ne 1 ]]; then
                continue
            fi

            # Parallel algos: skip P=1 (serial already covers it)
            if [[ "$ALGO" != "$ALGO_SER" && "$P" -eq 1 ]]; then
                continue
            fi

            # MM-2D requires perfect square P
            if [[ "$ALGO" == "$ALGO_2D" ]] && ! is_perfect_square "$P"; then
                echo "SKIP  algo=$ALGO  m=$M n=$N q=$Q  P=$P (not a perfect square)"
                continue
            fi

            # Warmup runs (not recorded)
            for (( w=0; w<WARMUP; w++ )); do
                mpirun --oversubscribe -np "$P" "$BINARY" "$ALGO" "$M" "$N" "$Q" "$P" > /dev/null 2>&1 || true
            done

            # Timed repetitions
            for (( rep=1; rep<=REPEATS; rep++ )); do
                if ! OUTPUT=$(mpirun --oversubscribe -np "$P" "$BINARY" "$ALGO" "$M" "$N" "$Q" "$P" 2>&1); then
                    echo "ERROR algo=$ALGO  m=$M n=$N q=$Q  P=$P  rep=$rep (non-zero exit)"
                    echo "${ALGO},${M},${N},${Q},${P},${rep},ERROR" >> "$OUTFILE"
                    continue
                fi

                TIME=$(extract_time "$OUTPUT")

                if [[ -z "$TIME" ]]; then
                    if [[ $TIMING_WARN_SHOWN -eq 0 ]]; then
                        echo ""
                        echo "WARN  No 'TIME: <float>' found in stdout."
                        echo "      Ask teammate to add this line at the end of main():"
                        echo "        printf(\"TIME: %.9f\\n\", t_end - t_start);"
                        echo "      where t_start/t_end use MPI_Wtime()."
                        echo "      (This warning shown once; further missing timings recorded as 0)"
                        echo ""
                        TIMING_WARN_SHOWN=1
                    fi
                    TIME="0"
                fi

                echo "${ALGO},${M},${N},${Q},${P},${rep},${TIME}" >> "$OUTFILE"
            done

            echo "DONE  algo=$ALGO  m=$M n=$N q=$Q  P=$P  (${REPEATS} reps)"

        done
    done
done

# ============================================================
# SUMMARY
# ============================================================
TOTAL_ROWS=$(( $(wc -l < "$OUTFILE") - 1 ))
ERROR_ROWS=$(grep -c ",ERROR$" "$OUTFILE" || true)

echo ""
echo "========================================"
echo " Sweep complete"
echo " CSV    : $OUTFILE"
echo " Rows   : $TOTAL_ROWS  (errors: $ERROR_ROWS)"
echo "========================================"
echo ""
echo "Next step: python3 scripts/analyze.py $OUTFILE"
