"""运行当前实验；参数通过--config指定。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import main

if __name__ == "__main__":
    main(default_experiment="table2")
