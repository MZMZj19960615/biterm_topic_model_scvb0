from typing import Any
import os
from pathlib import Path
import json
import pickle
import numpy as np

from ..get_caller_dir import get_caller_dir
from ..run_in_small_mode import RUN_IN_SMALL_MODE

DATA_DIR_NAME = 'DATA'
if RUN_IN_SMALL_MODE:
    DATA_DIR_NAME = 'DATA/.small'

def load_jsonl(filename):
    class JSONLReader:
        def __iter__(self):
            with open(filename) as f:
                for l in f:
                    yield json.loads(l[:-1])
    return JSONLReader()

def load_lines(filename):
    with open(filename) as f:
        for l in f:
            yield l[:-1]

class DefaultDirArg: pass
def load(filename, constructor=None, dir=DefaultDirArg()):
    '''
    Load `filename` in data directory.
    If *dir* is specified, `filename` is loaded from *dir* directory.
    '''
    ext = os.path.splitext(filename)[1]
    if dir is None:
        filename = str(filename)
    elif isinstance(dir, DefaultDirArg):
        filename = str(get_caller_dir() / DATA_DIR_NAME / filename)
    else:
        filename = str(get_caller_dir() / dir / DATA_DIR_NAME / filename)

    if ext == '.json':
        with open(filename) as f:
            return json.load(f)

    if ext == '.jsonl':
        return load_jsonl(filename)

    if ext == '.npy':
        return np.load(filename)

    if ext == '.pickle':
        with open(filename, 'rb') as f:
            return pickle.load(f)

    if ext == '' and constructor is not None and hasattr(constructor, 'load'):
        return constructor.load(filename)

    raise RuntimeError('unknown file type')

def get_data_dir(dir=None):
    if dir is None:
        return get_caller_dir() / DATA_DIR_NAME
    else:
        return get_caller_dir() / dir / DATA_DIR_NAME



def save_jsonl(filename, iterable):
    with open(filename, 'w') as f:
        for item in iterable:
            print(json.dumps(item, ensure_ascii=False, sort_keys=True), file=f)


def save(filename: str, obj: Any):
    '''
    save `obj` as `filename` in the data directory.
    The file type is determined by the extension of `filename`.
    '''
    filename = str(get_caller_dir() / DATA_DIR_NAME / filename)
    ext = os.path.splitext(filename)[1]
    out_dir = os.path.dirname(filename)
    os.makedirs(out_dir, exist_ok=True)

    if ext == '.pickle':
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
        return
    if ext == '.npy':
        np.save(filename, obj)
        return

    if ext == '.json':
        with open(filename, 'w') as f:
            json.dump(obj, f)
        return

    if ext == '.jsonl':
        save_jsonl(filename, obj)
        return

    if ext == '' and hasattr(obj, 'save'):
        obj.save(filename)
        return

    raise RuntimeError('unknown file type')
