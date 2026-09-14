"""pytest 路径引导：确保项目根在 sys.path，使 `from src import config` 可解析。"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
