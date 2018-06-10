import inspect
import os
import re
from pathlib import Path


def get_caller_dir() -> Path:
    filename = inspect.stack(context=0)[2].filename
    if re.fullmatch(r'<.*>', filename):
        raise RuntimeError(f'{filename} is not actual filename')

    dir_path = Path(os.path.dirname(os.path.abspath(filename)))
    return dir_path
