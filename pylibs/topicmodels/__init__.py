'''
Load topicmodels as Python modules
'''
import os
import subprocess

from my_lib import import_file
from my_lib import script_dir


def load_topic_model_class(name):
    mod = import_file(script_dir() / f'cpp_codes/{name.lower()}/bin/mod.so')
    return getattr(mod, name)

_PREV_DIR = os.getcwd()
os.chdir(script_dir() / 'cpp_codes')
subprocess.run('make', shell=True)
os.chdir(_PREV_DIR)

MODELS = ['BTM_SCVB0_V2','BTM_SCVB0', 'BTM_CGS', 'SPTM_CGS']

for model_name in MODELS:
    locals()[model_name] = load_topic_model_class(model_name)
