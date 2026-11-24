import hashlib
from pathlib import Path

import requests
from huggingface_hub import get_token, hf_hub_url
from tqdm import tqdm


filename = 'heco_final_siglip2_giant_checkpoint.ckpt'
path = Path('release_checkpoint') / filename
path.parent.mkdir(parents=True, exist_ok=True)
url = hf_hub_url(repo_id='dqj5182/heco-checkpoints', filename=filename, repo_type='dataset')
token = get_token()
headers = {'Authorization': f'Bearer {token}'} if token else {}
temporary = path.with_suffix(path.suffix + '.part')
digest = hashlib.sha256()


try:
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        total = int(response.headers.get('Content-Length', 0)) or None

        with temporary.open('wb') as stream, tqdm(total=total, unit='B', unit_scale=True, desc=filename) as progress:
            for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                if block:
                    stream.write(block)
                    digest.update(block)
                    progress.update(len(block))

    if digest.hexdigest() != '0e9616e47c0c7909d38eca2531f02f9c1ef10637ee372a8901ea39259dc13588':
        raise RuntimeError('Checkpoint checksum mismatch')

    temporary.replace(path)
finally:
    temporary.unlink(missing_ok=True)


print('Checkpoint saved to', path)