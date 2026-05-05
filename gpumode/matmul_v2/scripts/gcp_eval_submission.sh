#!/usr/bin/env bash
set -euo pipefail

INSTANCE="${INSTANCE:-cuda-l4-dev-lesson10}"
ZONE="${ZONE:-us-west1-b}"
MODE="${1:-benchmark}"
SUBMISSION="${2:-gpumode/matmul_v2/submissions/v0_safe.py}"

if [[ "$MODE" != "test" && "$MODE" != "benchmark" && "$MODE" != "leaderboard" ]]; then
  echo "usage: $0 {test|benchmark|leaderboard} path/to/submission.py" >&2
  exit 2
fi

REMOTE_ROOT="/home/xavier"
REMOTE_REPO="$REMOTE_ROOT/reference-kernels"
REMOTE_PROJ="$REMOTE_ROOT/cudatraining/gpumode/matmul_v2"
REMOTE_PROBLEM="$REMOTE_REPO/problems/pmpp_v2/matmul_py"
REMOTE_SUB="$REMOTE_PROJ/submissions/$(basename "$SUBMISSION")"

gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command \
  "mkdir -p '$REMOTE_PROJ/submissions' '$REMOTE_PROJ/scripts'"

gcloud compute scp "$SUBMISSION" "$INSTANCE:$REMOTE_SUB" --zone "$ZONE"

gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command "
set -euo pipefail
cd '$REMOTE_ROOT'
if [ ! -d '$REMOTE_REPO' ]; then
  git clone --depth 1 https://github.com/gpu-mode/reference-kernels.git '$REMOTE_REPO'
else
  cd '$REMOTE_REPO' && git pull --ff-only
fi
cd '$REMOTE_PROBLEM'
cp ../eval.py ../utils.py .
cp '$REMOTE_SUB' submission.py
cat > test_cases.txt <<'EOF'
m:64;n:64;k:64;seed:53124
m:128;n:128;k:128;seed:3321
m:256;n:256;k:256;seed:1200
m:32;n:512;k:32;seed:32523
m:64;n:1024;k:64;seed:4327
EOF
cat > benchmark_cases.txt <<'EOF'
m:128;n:128;k:128;seed:43214
m:256;n:256;k:256;seed:423011
m:512;n:512;k:512;seed:123456
m:1024;n:1024;k:1024;seed:1029
m:2048;n:2048;k:2048;seed:75342
m:1024;n:1536;k:1024;seed:321
m:2048;n:3072;k:2048;seed:32412
m:4096;n:5120;k:4096;seed:123456
EOF
CASES=test_cases.txt
if [ '$MODE' != test ]; then
  CASES=benchmark_cases.txt
fi
OUT='$(basename "$SUBMISSION" .py)'_'$MODE'.out
POPCORN_FD=3 python3 eval.py '$MODE' \"\$CASES\" 3> \"\$OUT\"
status=\$?
cat \"\$OUT\"
exit \$status
"
