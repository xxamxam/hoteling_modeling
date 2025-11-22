# hotelling_lib/__init__.py

from .mirror_descent import run_mirror_descent
from .rl_module import run_rl
from .compare import compare_modules

__all__ = [
    "run_mirror_descent",
    "run_rl",
    "compare_modules",
]
