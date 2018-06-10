from pathlib import Path
from .get_caller_dir import get_caller_dir


def script_dir() -> Path:
    '''Return the directory which contains the caller of this function
    '''
    return get_caller_dir()
