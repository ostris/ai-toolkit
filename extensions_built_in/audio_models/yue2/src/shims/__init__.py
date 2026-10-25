"""Stand-ins for mir_eval.chord and pretty_midi so m-a-p/SheetSage2's remote code imports without
those packages. Installed into sys.modules only when the real packages are absent."""

import importlib
import sys
import types


def install_shims():
    for name in ("mir_eval", "pretty_midi"):
        try:
            importlib.import_module(name)
            continue  # real package available; leave it alone
        except ImportError:
            pass
        if name == "mir_eval":
            from . import mir_eval_chord

            pkg = types.ModuleType("mir_eval")
            pkg.chord = mir_eval_chord
            pkg.__path__ = []
            sys.modules["mir_eval"] = pkg
            sys.modules["mir_eval.chord"] = mir_eval_chord
        else:
            from . import pretty_midi

            sys.modules["pretty_midi"] = pretty_midi
