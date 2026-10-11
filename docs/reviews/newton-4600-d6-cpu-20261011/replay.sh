#!/usr/bin/env bash
set -euo pipefail
: "${BASE_CHECKOUT:?Set BASE_CHECKOUT to a clean a6e1649b112e3d962b35f7dad66780dc55150588 checkout}"
: "${OWNER_CHECKOUT:?Set OWNER_CHECKOUT to a clean c7a91b485e33ec489684c79b0deda2dd121c439f checkout}"
: "${INTERPRETER:?Set INTERPRETER to an existing Python interpreter with the dependencies installed}"
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_CHECKOUT=$(cd -- "$BASE_CHECKOUT" && pwd)
OWNER_CHECKOUT=$(cd -- "$OWNER_CHECKOUT" && pwd)
RESULTS_DIR=${RESULTS_DIR:-"$PWD/newton-4600-results"}
mkdir -p -- "$RESULTS_DIR"
RESULTS_DIR=$(cd -- "$RESULTS_DIR" && pwd)
export GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 UV_OFFLINE=1
command -v uv >/dev/null
check_source() {
  local checkout=$1 expected=$2
  [[ $(git -C "$checkout" rev-parse HEAD) == "$expected" ]] || { echo 'Wrong checkout commit' >&2; exit 2; }
  [[ -z $(git -C "$checkout" status --porcelain=v1) ]] || { echo 'Source checkout must be clean' >&2; exit 2; }
}
check_source "$BASE_CHECKOUT" a6e1649b112e3d962b35f7dad66780dc55150588
check_source "$OWNER_CHECKOUT" c7a91b485e33ec489684c79b0deda2dd121c439f
run_gate() {
  local label=$1 expected=$2 checkout=$3 status
  shift 3
  set +e
  (cd -- "$checkout"; PYTHONPATH="$checkout" uv run --offline --no-project --python "$INTERPRETER" python "$HERE/run_verified.py" "$checkout" "$@") > "$RESULTS_DIR/$label.log" 2>&1
  status=$?
  set -e
  printf '%s exit=%s expected=%s\n' "$label" "$status" "$expected"
  [[ "$status" -eq "$expected" ]] || { cat "$RESULTS_DIR/$label.log"; return 1; }
}
run_gate baseline-witness 1 "$BASE_CHECKOUT" script "$HERE/witness.py"
run_gate owner-witness 0 "$OWNER_CHECKOUT" script "$HERE/witness.py"
run_gate baseline-physical 1 "$BASE_CHECKOUT" script "$HERE/physical_probe.py" --expected-root "$BASE_CHECKOUT"
run_gate owner-physical 0 "$OWNER_CHECKOUT" script "$HERE/physical_probe.py" --expected-root "$OWNER_CHECKOUT"
run_gate owner-focused 0 "$OWNER_CHECKOUT" unittest newton.tests.test_joint_drive -k test_semi_implicit_d6_2dof_drive_at_target -v
run_gate owner-adjacent 0 "$OWNER_CHECKOUT" unittest newton.tests.test_joint_drive newton.tests.test_joint_limits -q
# Check expected case counts too, so an unrelated baseline failure cannot pass.
[[ $(grep -cE '^[XYZ] [XYZ]: .* PASS$' "$RESULTS_DIR/baseline-witness.log") == 1 ]]
[[ $(grep -cE '^[XYZ] [XYZ]: .* FAIL$' "$RESULTS_DIR/baseline-witness.log") == 5 ]]
[[ $(grep -cE '^[XYZ] [XYZ]: .* PASS$' "$RESULTS_DIR/owner-witness.log") == 6 ]]
grep -q '^SUMMARY 3 PASS 12 FAIL ATOL 1e-08 POSE_ATOL 1e-06$' "$RESULTS_DIR/baseline-physical.log"
grep -q '^SUMMARY 15 PASS 0 FAIL ATOL 1e-08 POSE_ATOL 1e-06$' "$RESULTS_DIR/owner-physical.log"
grep -q '^Ran 1 test ' "$RESULTS_DIR/owner-focused.log"
grep -q '^Ran 26 tests ' "$RESULTS_DIR/owner-adjacent.log"
echo 'All expected CPU regression gates reproduced.'
