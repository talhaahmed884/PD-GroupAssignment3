#!/usr/bin/env bash
# run_correctness.sh
# Validates all algorithm implementations against the serial reference.
# Run locally:  bash scripts/run_correctness.sh
# Run via WSL:  bash scripts/run_wsl.sh run_correctness.sh
#
# Requires: ./mm and ./correctness_check to be built (run `make && make check-build`)
# Requires: mpirun to be available in PATH

set -euo pipefail

BINARY="${BINARY:-./mm}"
CHECKER="${CHECKER:-./correctness_check}"

# Tell mm.cpp to print the result matrix so the checker can validate it
export MM_PRINT_MATRIX=1

# ============================================================
# CHECK BINARIES EXIST
# ============================================================
if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: Binary '$BINARY' not found. Run: bash scripts/run_wsl.sh make"
    exit 1
fi
if [[ ! -f "$CHECKER" ]]; then
    echo "ERROR: Correctness checker '$CHECKER' not found. Run: bash scripts/run_wsl.sh make check-build"
    exit 1
fi

# ============================================================
# CONFIG
# ============================================================
# Algorithm selector strings
ALGO_SER="${ALGO_SER:-ser}"
ALGO_1D="${ALGO_1D:-1d}"
ALGO_2D="${ALGO_2D:-2d}"

# P=4 is a perfect square, works for all algorithms including MM-2D
P_TEST=4

# Small matrix sizes: "m n q"
SIZES=("32 32 32" "64 32 64" "128 64 128")

PASS=0
FAIL=0
SKIP=0

echo "========================================"
echo " Correctness Checks"
echo " Binary : $BINARY"
echo " Checker: $CHECKER"
echo " P      : $P_TEST (perfect square)"
echo "========================================"
echo ""

# ============================================================
# MAIN LOOP
# ============================================================
for SIZE in "${SIZES[@]}"; do
    read -r M N Q <<< "$SIZE"
    echo "--- Matrix ${M}x${N}x${Q} ---"

    for ALGO in "$ALGO_SER" "$ALGO_1D" "$ALGO_2D"; do
        # Serial uses P=1; parallel algos use P_TEST
        if [[ "$ALGO" == "$ALGO_SER" ]]; then
            P=1
        else
            P=$P_TEST
        fi

        # Run via mpirun — required for MPI binary regardless of P value
        if ! OUTPUT=$(mpirun -np "$P" "$BINARY" "$ALGO" "$M" "$N" "$Q" "$P" 2>&1); then
            echo "  SKIP : algo=$ALGO (binary returned non-zero — may not be implemented yet)"
            ((SKIP++)) || true
            continue
        fi

        # Check that timing is present; warn but don't fail
        if ! echo "$OUTPUT" | grep -q "TIME:"; then
            echo "  WARN : algo=$ALGO — no 'TIME: <float>' found in stdout."
            echo "         Ask teammate to add: printf(\"TIME: %.9f\\n\", t_end - t_start);"
        fi

        # Pipe output to correctness checker
        if echo "$OUTPUT" | "$CHECKER" "$M" "$N" "$Q"; then
            echo "  PASS : algo=$ALGO  m=$M n=$N q=$Q  P=$P"
            ((PASS++)) || true
        else
            echo "  FAIL : algo=$ALGO  m=$M n=$N q=$Q  P=$P"
            ((FAIL++)) || true
        fi
    done
    echo ""
done

# ============================================================
# SUMMARY
# ============================================================
echo "========================================"
echo " Results: $PASS passed | $FAIL failed | $SKIP skipped"
echo "========================================"

if [[ $FAIL -gt 0 ]]; then
    echo "ACTION: Fix failing implementations before running timing experiments."
    echo "        Speedup numbers from incorrect code are meaningless."
    exit 1
fi

if [[ $SKIP -gt 0 ]]; then
    echo "NOTE: $SKIP algorithm(s) were skipped (not yet implemented)."
    echo "      Re-run after all teammates submit their code."
fi

exit 0
