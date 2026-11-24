import argparse
import getpass
from pathlib import Path
import tarfile
import zipfile

import requests
from tqdm import tqdm


def download(url, destination, username, password):
    partial = destination.with_suffix(destination.suffix + '.part')
    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.post(url, data={'username': username, 'password': password}, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()

        if 'text/html' in response.headers.get('Content-Type', '').lower():
            raise RuntimeError('Download failed. Check your account credentials and dataset access.')

        total = int(response.headers.get('Content-Length', 0))

        with partial.open('wb') as stream, tqdm(total=total, unit='B', unit_scale=True, desc=destination.name) as progress:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    stream.write(chunk)
                    progress.update(len(chunk))

    if destination.suffix == '.zip':
        with zipfile.ZipFile(partial) as archive:
            if archive.testzip() is not None:
                raise RuntimeError('Downloaded ZIP archive failed validation')
    else:
        with tarfile.open(partial, 'r:gz') as archive:
            archive.getmembers()

    partial.replace(destination)


def extract(path, destination):
    destination.mkdir(parents=True, exist_ok=True)

    root = destination.resolve()

    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not (root / name).resolve().is_relative_to(root):
                    raise ValueError('Invalid archive path')

            members = [name for name in archive.namelist()
                       if '__MACOSX' not in Path(name).parts and not Path(name).name.startswith('._')]
            archive.extractall(destination, members=members)
    else:
        with tarfile.open(path, 'r:gz') as archive:
            for member in archive.getmembers():
                if not (root / member.name).resolve().is_relative_to(root):
                    raise ValueError('Invalid archive path')

                if member.issym() or member.islnk():
                    target = root / member.name
                    link = target.parent / member.linkname if member.issym() else root / member.linkname

                    if not link.resolve().is_relative_to(root):
                        raise ValueError('Invalid archive link')

            archive.extractall(destination)


def credentials(name):
    print('Enter your {} download credentials.'.format(name))

    return input('Username: '), getpass.getpass('Password: ')


parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path('data/DAMON/data'))
parser.add_argument('--dataset', choices=['DAMON', 'HOT'], default='DAMON')
args = parser.parse_args()


root = args.output

if args.dataset == 'DAMON':
    name, domain = 'DAMON/DECO', 'deco'
    tasks = [('Release_Damon.tar.gz', root),
             ('damon_segmentations.tar.gz', root / 'HOT-Annotated')]
else:
    name, domain = 'HOT', 'hot'
    tasks = [('HOT-Annotated.zip', root)]


username, password = credentials(name)

for filename, destination in tasks:
    url = 'https://download.is.tue.mpg.de/download.php?domain={}&resume=1&sfile={}'.format(domain, filename)
    path = root / filename
    download(url, path, username, password)
    extract(path, destination)
    path.unlink()


print(args.dataset, 'saved to', root)