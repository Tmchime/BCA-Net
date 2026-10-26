"""C14/C15：运行当前实验，--config指定配置文件。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run import main

if __name__ == "__main__":
    main(default_experiment="student_image")
