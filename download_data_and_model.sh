#!/usr/bin/env bash
# Public Zenodo archives; DATA_URL and MODEL_URL may override these locations.
set -euo pipefail
cd "$(dirname "$0")"
DATA_URL=${DATA_URL:-https://zenodo.org/records/22720303/files/data.tar.gz?download=1}
MODEL_URL=${MODEL_URL:-https://zenodo.org/records/22720378/files/AutogluonModels.tar.gz?download=1}

for target in data AutogluonModels; do
    if [[ -e "$target" ]]; then
        echo "Refusing to overwrite $target. Download archives into a clean project copy." >&2
        exit 1
    fi
done

# Temporary archives are removed automatically; extraction permits regular
# files/directories only, inside the expected top-level directory.
python - "$DATA_URL" "$MODEL_URL" <<'PY'
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from urllib.request import urlopen

with tempfile.TemporaryDirectory(prefix='cosork-download-') as temporary:
    staging = Path(temporary)
    for target, url in zip(('data', 'AutogluonModels'), sys.argv[1:]):
        print(f'Downloading {target}...', flush=True)
        archive = staging / f'{target}.tar.gz'
        with urlopen(url, timeout=60) as response, archive.open('wb') as out:
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' in content_type:
                raise RuntimeError(f'{target}: server returned an HTML/login page, not an archive')
            shutil.copyfileobj(response, out)
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            for member in members:
                path = Path(member.name)
                if (path.is_absolute() or '..' in path.parts or not path.parts
                        or path.parts[0] != target or not (member.isfile() or member.isdir())):
                    raise ValueError(f'Unexpected archive member: {member.name}')
            tar.extractall(staging, members=members)
        if not (staging / target).is_dir():
            raise ValueError(f'Archive does not contain {target}/')
    # Publish only after BOTH downloads have been validated.
    for target in ('data', 'AutogluonModels'):
        if Path(target).exists():
            raise FileExistsError(target)
        shutil.move(str(staging / target), target)
print('Downloaded data/ and AutogluonModels/.')
PY
