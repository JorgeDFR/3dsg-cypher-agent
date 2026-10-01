# ruff: noqa: E402
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402, F401

from cypher_3dsg_agent_distillation.cypher.templates import main

if __name__ == "__main__":
    main()
