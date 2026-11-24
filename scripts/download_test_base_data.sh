set -euo pipefail

TARGET_DIR="data"
ARCHIVE_NAME="$TARGET_DIR/base_data.tar.gz"

mkdir -p "$TARGET_DIR"


echo "Downloading base_data.tar.gz..."
python - <<'PY'
from shutil import copyfile
from huggingface_hub import hf_hub_download


path = hf_hub_download(
    repo_id='dqj5182/heco-data',
    filename='eval/data/base_data.tar.gz',
    repo_type='dataset',
)

copyfile(path, 'data/base_data.tar.gz')
PY


echo "Decompressing into $TARGET_DIR..."
tar -xvzf "$ARCHIVE_NAME" -C "$TARGET_DIR"
rm "$ARCHIVE_NAME"
echo "Done. Extracted to $TARGET_DIR"