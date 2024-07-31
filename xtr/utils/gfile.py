import os
import urllib.request

XTR_MODEL_CACHE_DIR = "~/.cache/xtr"

def _gfile_local_path(path: str):
    assert path.startswith("gs://")
    filename = path[len("gs://"):].replace("/", ".")
    return os.path.join(os.path.expanduser(XTR_MODEL_CACHE_DIR), "gfile", filename)

def _gfile_public_url(path: str):
    assert path.startswith("gs://")
    filename = path[len("gs://"):]
    return f"https://storage.googleapis.com/{filename}"

def gfile_load(path: str):
    filename = _gfile_local_path(path)
    if not os.path.exists(filename):
        os.makedirs(os.path.join(os.path.expanduser(XTR_MODEL_CACHE_DIR), "gfile"), exist_ok=True)
        print(f"Downloading {path}.", end="")
        with open(filename, "wb") as file:
            with urllib.request.urlopen(_gfile_public_url(path)) as gfile:
                data = gfile.read()
                file.write(data)
    else:
        print(f"Loading cached {path}.", end="")
        with open(filename, "rb") as file:
            data = file.read()
    print(" Done")
    return data