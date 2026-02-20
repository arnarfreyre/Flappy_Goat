"""
Monkey-patch FlappyBird._load_images to suppress libpng C-level warnings.

Usage (in any notebook, after importing FlappyBird):
    from silence_libpng import patch_flappy
    patch_flappy(FlappyBird)

    game = FlappyBird()  # no libpng warnings
"""

import os


def _suppress_c_stderr(func):
    """Wrap a function so C-level stderr is silenced during its execution."""
    def wrapper(*args, **kwargs):
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_fd = os.dup(2)
        os.dup2(devnull, 2)
        try:
            return func(*args, **kwargs)
        finally:
            os.dup2(old_fd, 2)
            os.close(devnull)
            os.close(old_fd)
    return wrapper


def patch_flappy(flappy_cls):
    """Patch FlappyBird so _load_images silences libpng warnings."""
    if getattr(flappy_cls, '_libpng_patched', False):
        return
    flappy_cls._load_images = _suppress_c_stderr(flappy_cls._load_images)
    flappy_cls._libpng_patched = True
