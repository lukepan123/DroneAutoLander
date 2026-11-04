#!/usr/bin/env bash
# =============================================================================
#  run_conditions.sh (SWEEP-TEMP) : the four measurement-model conditions, back to back
#
#    mesh_true      projected mesh, true pad height
#    raycast_true   ray-cast baseline, true pad height
#    mesh_wrong     projected mesh, pad height assumed 3.0 m
#    raycast_wrong  ray-cast baseline, pad height assumed 3.0 m
#
#  Each lands in sweep_out/<tag>/. Re-running resumes each condition where it stopped.
#  Usage : ./run_conditions.sh            all four, full sweep
#          ./run_conditions.sh --limit 3  smoke test of all four
#  Delete this file after testing.
# =============================================================================
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRONG_HEIGHT=3.0

# Never inherit a stale condition from the calling shell
unset AUTO_LANDER_RAYCAST AUTO_LANDER_ROOF_INIT SWEEP_TAG

# Both methods get the true car heading (the filter's yaw state in flight). Set
# TRUE_YAW=0 ./run_conditions.sh to repeat the old assumed-heading sweep.
export AUTO_LANDER_TRUE_YAW=${TRUE_YAW:-1}

run() {     # $1 = tag, $2 = raycast (0/1), $3 = roof height or "" for true; rest -> sweep_runner
    local tag=$1 rc=$2 roof=$3; shift 3
    echo "================ $tag ================"
    if [[ -n $roof ]]; then
        SWEEP_TAG=$tag AUTO_LANDER_RAYCAST=$rc AUTO_LANDER_ROOF_INIT=$roof "$HERE/run_sweep.sh" "$@" \
            || echo "!!! $tag exited with an error, continuing"
    else
        SWEEP_TAG=$tag AUTO_LANDER_RAYCAST=$rc "$HERE/run_sweep.sh" "$@" \
            || echo "!!! $tag exited with an error, continuing"
    fi
}

run mesh_true     0 ""            "$@"
run raycast_true  1 ""            "$@"
run mesh_wrong    0 $WRONG_HEIGHT "$@"
run raycast_wrong 1 $WRONG_HEIGHT "$@"

echo "Done. Results in $HERE/sweep_out/{mesh_true,raycast_true,mesh_wrong,raycast_wrong}"
