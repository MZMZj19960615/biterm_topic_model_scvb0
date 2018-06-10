import os
import importlib.machinery


def import_file(filename: str):
    '''Import *filename* as a Python module, dynamically.
    '''
    filename = str(filename)
    module_name, ext = os.path.splitext(filename)
    module_name = module_name.replace('/', '.')
    if ext == '.py':
        module = importlib.machinery.SourceFileLoader(module_name, filename).load_module()
    elif ext == '.so':
        module = importlib.machinery.ExtensionFileLoader(module_name, filename).load_module()
    else:
        raise RuntimeError('import_file error')

    return module
