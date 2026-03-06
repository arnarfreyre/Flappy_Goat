#!/usr/bin/env python
"""
Doom Import Diagnostic & Launcher

Diagnoses why 'Couldn't import doom' prints on every PLE import,
and attempts to run Doom if all dependencies are present.

Import chain:
  ple/games/__init__.py
    -> ple/games/doom/__init__.py
      -> doom.py
        -> base/doomwrapper.py
          -> doom_py (external C++ ViZDoom bindings)
          -> numpy
          -> pygame
"""
import sys
import os
import importlib

SEPARATOR = "-" * 50


def check_dependency(name, import_name=None, extra_info=""):
    """Check if a Python package is importable. Returns (ok, version_or_error)."""
    import_name = import_name or name
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", getattr(mod, "VERSION", "unknown"))
        return True, version
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_doom_py_submodules():
    """If doom_py imports, check its critical submodule."""
    try:
        import doom_py.vizdoom as vizdoom
        return True, "vizdoom submodule OK"
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_doom_class():
    """Try importing the actual Doom class from the project."""
    # Add project root to path so relative imports resolve
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from ple.games.doom.doom import Doom
        return True, "Doom class imported successfully"
    except ImportError as e:
        return False, f"ImportError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def print_result(label, ok, detail):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    print(f"         {detail}")


def main():
    print()
    print("Doom Import Diagnostic")
    print(SEPARATOR)
    print()

    # 1. Check individual dependencies
    deps = [
        ("numpy", "numpy", "pip install numpy"),
        ("pygame", "pygame", "pip install pygame"),
        ("doom_py", "doom_py", "pip install doom_py  (requires C++ build tools)"),
    ]

    all_ok = True
    missing = []

    print("1) Dependency Check")
    print()
    for name, import_name, install_hint in deps:
        ok, detail = check_dependency(name, import_name)
        if ok:
            print_result(name, True, f"version {detail}")
        else:
            print_result(name, False, detail)
            print(f"         Install: {install_hint}")
            missing.append(name)
            all_ok = False
    print()

    # 2. Check doom_py.vizdoom submodule (only if doom_py itself imported)
    print("2) doom_py.vizdoom Submodule")
    print()
    if "doom_py" not in [m[0] for m in deps if m[0] in missing]:
        ok, detail = check_doom_py_submodules()
        print_result("doom_py.vizdoom", ok, detail)
        if not ok:
            all_ok = False
    else:
        print("  [SKIP] doom_py not installed, skipping submodule check")
    print()

    # 3. Try importing the Doom class
    print("3) Doom Class Import")
    print()
    ok, detail = check_doom_class()
    print_result("from ple.games.doom.doom import Doom", ok, detail)
    if not ok:
        all_ok = False
    print()

    # 4. Available scenarios
    print("4) Available Scenarios")
    print()
    assets_dir = os.path.join(os.path.dirname(__file__), "assets", "cfg")
    if os.path.isdir(assets_dir):
        cfgs = sorted(f[:-4] for f in os.listdir(assets_dir) if f.endswith(".cfg"))
        for cfg in cfgs:
            print(f"    - {cfg}")
    else:
        print("  [WARN] assets/cfg directory not found")
    print()

    # 5. Summary
    print(SEPARATOR)
    if all_ok:
        print("All checks passed. Doom should be usable.")
        print()
        print("To launch a basic scenario:")
        print("  from ple.games.doom.doom import Doom")
        print('  doom = Doom(scenario="basic")')
    else:
        print("Doom is NOT usable. Missing/broken dependencies:")
        if missing:
            print()
            print("  Install missing packages:")
            for name, _, hint in deps:
                if name in missing:
                    print(f"    {hint}")
        print()
        print("Note: doom_py requires ViZDoom's C++ engine compiled from")
        print("source. See: https://github.com/mwydmuch/ViZDoom")
        print()
        print("This is why 'Couldn't import doom' prints on every PLE import.")
        print("The bare except in ple/games/__init__.py silently catches this.")
    print()


if __name__ == "__main__":
    main()
