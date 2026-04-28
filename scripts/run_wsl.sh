#!/usr/bin/env bash
# run_wsl.sh
# Windows-side entry point. Handles the parentheses in the repo path that break
# shell expansion, and routes all script execution into WSL.
#
# Usage (from Windows Git Bash):
#   bash scripts/run_wsl.sh run_experiments.sh
#   bash scripts/run_wsl.sh run_correctness.sh
#   bash scripts/run_wsl.sh make          (build inside WSL)
#
# To compile only:
#   bash scripts/run_wsl.sh make
#   bash scripts/run_wsl.sh make clean

REPO_WSL="/mnt/c/Users/<YOUR_WINDOWS_USERNAME>/path/to/GroupAssignment3"  # TODO: update to your WSL path

if [[ $# -eq 0 ]]; then
    echo "Usage: bash scripts/run_wsl.sh <script_or_command> [args...]"
    echo "Examples:"
    echo "  bash scripts/run_wsl.sh make"
    echo "  bash scripts/run_wsl.sh run_experiments.sh"
    echo "  bash scripts/run_wsl.sh run_correctness.sh"
    exit 1
fi

SUBCMD="$1"
shift

# Determine if this is a make command or a script
if [[ "$SUBCMD" == "make" ]]; then
    wsl bash -c "cd '${REPO_WSL}' && make $*"
else
    wsl bash -c "cd '${REPO_WSL}' && bash scripts/${SUBCMD} $*"
fi
