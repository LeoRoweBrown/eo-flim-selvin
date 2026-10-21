"""Generate a Pylance/pyright stub for pycromanager's Core() object.

Run this once with Micro-Manager open (and the ZMQ server enabled), from the
folder your own scripts live in. It writes mmcore_types.py there, a typing-only
Protocol you can import and cast Core() to:

    from mmcore_types import CMMCore
    core = cast(CMMCore, Core())
"""
import inspect
import keyword
from pathlib import Path

from pycromanager import Core

OUT = Path("mmcore_types.py")


def ann(a) -> str:
    if a is inspect.Parameter.empty or a is inspect.Signature.empty or a is None:
        return "Any"
    if isinstance(a, type):
        if a.__module__ == "builtins":
            return a.__name__
        if a.__module__.startswith("numpy"):
            return "np.ndarray"
    return "Any"  # Java objects (TaggedImage, Configuration, ...) etc.


def method_stub(name: str, fn) -> str:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return f"    def {name}(self, *args: Any) -> Any: ..."
    parts = ["self"]
    for p in sig.parameters.values():
        pname = p.name + "_" if keyword.iskeyword(p.name) else p.name
        if p.kind is p.VAR_POSITIONAL:
            parts.append(f"*{pname}: Any")
        elif p.kind is p.VAR_KEYWORD:
            parts.append(f"**{pname}: Any")
        else:
            t = ann(p.annotation)
            if p.default is p.empty:
                parts.append(f"{pname}: {t}")
            elif t == "Any":
                parts.append(f"{pname}: Any = None")
            else:
                parts.append(f"{pname}: {t} | None = None")
    return f"    def {name}({', '.join(parts)}) -> {ann(sig.return_annotation)}: ..."


def main() -> None:
    core = Core()
    lines = [
        "from typing import Any, Protocol",
        "import numpy as np",
        "",
        "class CMMCore(Protocol):",
    ]
    for name in sorted(dir(core)):
        if name.startswith("_"):
            continue
        attr = getattr(core, name, None)
        if callable(attr):
            lines.append(method_stub(name, attr))
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines) - 4} methods to {OUT}")


if __name__ == "__main__":
    main()
