#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_env.sh — build the two conda envs the new-go infosel pipeline needs.
#
#   ENV_BUILD (default: code-attribute) — build + LoRA train + eval.
#       Must already have the repo training stack (torch + transformers + peft).
#       We add the AI4Go metric deps here: fuzzywuzzy python-Levenshtein rouge nltk pandas.
#   ENV_VLLM  (default: vllm)           — teacher-NLL labeling ONLY (vLLM).
#       Isolated on purpose: `pip install vllm` pins torch/transformers and would
#       otherwise downgrade/break the training env.
#
# Usage:  bash scripts/newgo_infosel/setup_env.sh
#         ENV_BUILD=myenv ENV_VLLM=myvllm bash scripts/newgo_infosel/setup_env.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
CONDA=${CONDA:-/home/v-murongma/miniconda3}
ENV_BUILD=${ENV_BUILD:-code-attribute}
ENV_VLLM=${ENV_VLLM:-vllm}
PYVER=${PYVER:-3.12}
source "$CONDA/etc/profile.d/conda.sh"

echo "=== [1/2] AI4Go eval-metric deps -> $ENV_BUILD ==="
# NB: use `python3 -m pip` (a stray ~/.local/bin/pip can shadow the env pip).
conda run -n "$ENV_BUILD" python3 -m pip install -q fuzzywuzzy python-Levenshtein rouge nltk pandas
conda run -n "$ENV_BUILD" python3 -c "import fuzzywuzzy,rouge,nltk,pandas,Levenshtein;print('  eval deps OK in',__import__('os').environ.get('CONDA_DEFAULT_ENV','?'))"

echo "=== [2/2] dedicated vLLM env: $ENV_VLLM ==="
if conda env list | awk '{print $1}' | grep -qx "$ENV_VLLM"; then
  echo "  $ENV_VLLM already exists"
else
  echo "  creating $ENV_VLLM (python $PYVER)"
  conda create -n "$ENV_VLLM" "python=$PYVER" -y
fi
conda run -n "$ENV_VLLM" python3 -m pip install -q vllm
conda run -n "$ENV_VLLM" python3 -c "import vllm,torch;print('  vllm',vllm.__version__,'| torch',torch.__version__,torch.version.cuda,'| cuda',torch.cuda.is_available())"

echo "=== DONE. envs ready: build=$ENV_BUILD  vllm=$ENV_VLLM ==="
