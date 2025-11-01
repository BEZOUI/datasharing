"""Entry-point to launch the interactive optimisation dashboard."""
from __future__ import annotations

from visualization.dashboard import launch_dashboard, tkinter_available


def main() -> None:
    if not tkinter_available():
        raise SystemExit(
            "Tkinter is not available in this environment. Install tkinter to use the dashboard interface."
        )
    launch_dashboard()


if __name__ == "__main__":
    main()
