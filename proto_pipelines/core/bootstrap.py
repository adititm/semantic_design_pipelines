"""Resolve the ``proto_language`` / ``proto_tools`` packages before first import.

In a clean environment ``pip install git+https://github.com/evo-design/proto-language``
makes both packages importable and this module is a no-op. On a workstation that
carries several checkouts (and possibly a stale editable install pointing at the
wrong one), set ``PROTO_LANGUAGE_ROOT`` to the checkout you want and import this
module first::

    export PROTO_LANGUAGE_ROOT=/path/to/proto-language

The checkout's bundled ``proto-tools`` submodule is used unless
``PROTO_TOOLS_ROOT`` overrides it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_APPLIED = False


def activate() -> None:
    """Prepend ``PROTO_LANGUAGE_ROOT`` (and its proto-tools) to ``sys.path``.

    Editable-install finders live on ``sys.meta_path`` and take precedence over
    ``sys.path``, so they are removed first; otherwise the env var would be
    silently ignored in favour of whatever the editable install points at.
    Does nothing when ``PROTO_LANGUAGE_ROOT`` is unset.
    """
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    root = os.environ.get("PROTO_LANGUAGE_ROOT")
    if not root:
        return

    root_path = Path(root).resolve()
    if not (root_path / "proto_language").is_dir():
        raise FileNotFoundError(
            f"PROTO_LANGUAGE_ROOT={root_path} does not contain a 'proto_language' package"
        )

    if "proto_language" in sys.modules or "proto_tools" in sys.modules:
        raise RuntimeError(
            "proto_language/proto_tools were imported before bootstrap.activate(); "
            "import proto_pipelines.core.bootstrap first."
        )

    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if "editable" not in getattr(finder, "__name__", "").lower()
        and "editable" not in type(finder).__module__.lower()
    ]

    tools_root = os.environ.get("PROTO_TOOLS_ROOT") or str(root_path / "proto-tools")
    tools_path = Path(tools_root).resolve()
    if not (tools_path / "proto_tools").is_dir():
        raise FileNotFoundError(
            f"proto-tools not found at {tools_path}; set PROTO_TOOLS_ROOT explicitly"
        )

    sys.path.insert(0, str(tools_path))
    sys.path.insert(0, str(root_path))


activate()
