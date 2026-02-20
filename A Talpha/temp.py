"""
Validate that silence_libpng.patch_flappy() correctly silences the
'libpng warning: iCCP: known incorrect sRGB profile' when creating
FlappyBird instances — the same way notebooks will use it.
"""

import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Each test runs in a subprocess for a clean pygame/FlappyBird import.
# ---------------------------------------------------------------------------

UNPATCHED = """
import os, sys
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
sys.path.insert(0, 'itml-project2')
from ple.games.flappybird import FlappyBird
game = FlappyBird()
print("GAME_OK")
"""

PATCHED = """
import os, sys
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
sys.path.insert(0, 'itml-project2')
from ple.games.flappybird import FlappyBird

# --- the one-liner notebooks will use ---
from silence_libpng import patch_flappy
patch_flappy(FlappyBird)

game = FlappyBird()
print("GAME_OK")
"""

PATCHED_MULTI = """
import os, sys
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
sys.path.insert(0, 'itml-project2')
from ple.games.flappybird import FlappyBird
from silence_libpng import patch_flappy
patch_flappy(FlappyBird)

# Multiple FlappyBird instances (training + greedy_run pattern)
g1 = FlappyBird()
g2 = FlappyBird()
g3 = FlappyBird()
print("GAME_OK")
"""

PATCHED_IDEMPOTENT = """
import os, sys
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
sys.path.insert(0, 'itml-project2')
from ple.games.flappybird import FlappyBird
from silence_libpng import patch_flappy
patch_flappy(FlappyBird)
patch_flappy(FlappyBird)  # double-patch should be safe
game = FlappyBird()
print("GAME_OK")
"""


def run_test(label, code):
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        cwd=PROJECT_ROOT,
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    game_ok = "GAME_OK" in stdout
    has_libpng = "libpng warning" in stderr

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Game created OK : {'yes' if game_ok else 'NO'}")
    print(f"  libpng warning  : {'PRESENT' if has_libpng else 'absent'}")
    if has_libpng:
        for line in stderr.splitlines():
            if "libpng" in line:
                print(f"    -> {line}")
    return game_ok, has_libpng


if __name__ == "__main__":
    results = []

    ok, warn = run_test("TEST 1: Unpatched (expect warning)", UNPATCHED)
    results.append(("Unpatched has warning", ok and warn))

    ok, warn = run_test("TEST 2: Patched (expect NO warning)", PATCHED)
    results.append(("Patched silences warning", ok and not warn))

    ok, warn = run_test("TEST 3: Multiple instances (expect NO warning)", PATCHED_MULTI)
    results.append(("Multiple instances silenced", ok and not warn))

    ok, warn = run_test("TEST 4: Idempotent patch (expect NO warning)", PATCHED_IDEMPOTENT)
    results.append(("Double-patch safe", ok and not warn))

    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  All tests passed.")
    else:
        print("  Some tests FAILED.")

    sys.exit(0 if all_pass else 1)
