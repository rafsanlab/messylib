import psutil
import os
import sys
from pympler import asizeof

_process = psutil.Process(os.getpid())


def log_memory(tag=""):
    """Log total memory usage of the current Python process."""
    mem_in_mb = _process.memory_info().rss / 1024**2
    print(f"{tag} Memory usage: {mem_in_mb:.2f} MB")


def sizeof(var, name=None, deep=False):
    """
    Print memory size of a single variable.

    Parameters:
    - var: The variable to inspect
    - name: Optional name to display
    """
    label = name or repr(var)[:30]
    if deep:
        size = asizeof.asizeof(var)
    else:
        size = sys.getsizeof(var)

    size_kb = size / 1024
    print(f"{label}: {size_kb:.2f} KB ({'deep' if deep else 'shallow'})")
