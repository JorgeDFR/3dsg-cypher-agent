from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cypher_3dsg_agent_distillation.core.cli_output import configure_cli_output  # noqa: E402

configure_cli_output()
