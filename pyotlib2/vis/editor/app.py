"""Interactive point set editor using PySide6.

Launch via:
    pyotlib2 editor [file]
or programmatically:
    from pyotlib2.vis.editor.app import run_editor
    run_editor("file.b08", n=6)

Features
--------
- Drag points with the mouse (integer coordinates, kept exactly as loaded)
- Info panel: coordinates, SmallLambda rank matrix and checksum
- OT lock: drags that would change the order type are refused
- Convex hull / onion layer overlay, point labels, collinear points in red
- Live properties, optimization (minimize / beautify / property walk)
- Save current configuration (File > Save)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def run_editor(
    filepath: Optional[str] = None,
    *,
    n: Optional[int] = None,
    fmt: Optional[str] = None,
) -> None:
    """Launch the interactive editor.  Blocks until the window is closed."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "Error: PySide6 is required for the editor.\n"
            "Install with:  pip install -e '.[vis]'",
            file=sys.stderr,
        )
        sys.exit(1)

    import signal
    from PySide6.QtCore import QTimer

    app = QApplication.instance() or QApplication(sys.argv)

    # Let Python handle SIGINT (Ctrl+C) by restoring the default handler,
    # then quit the Qt event loop cleanly.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())

    # Qt blocks Python signal delivery while in the C event loop.
    # A short timer forces a return to Python every 200ms so signals are processed.
    timer = QTimer()
    timer.start(200)
    timer.timeout.connect(lambda: None)

    from pyotlib2.vis.editor._window import EditorWindow
    win = EditorWindow(filepath=filepath, n=n, fmt=fmt)
    win.show()
    sys.exit(app.exec())
