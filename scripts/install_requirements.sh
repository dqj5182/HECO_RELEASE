set -euo pipefail

python -m pip install "setuptools<81" wheel ninja

CUDA_VISIBLE_DEVICES="" MAX_JOBS="${MAX_JOBS:-2}" python -m pip install --no-build-isolation -r "$(dirname "$0")/../requirements.txt"