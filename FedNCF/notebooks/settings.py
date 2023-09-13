from pathlib import Path
import pyrootutils

root = Path(__file__).parent.parent
pyrootutils.set_root(path=root, pythonpath=True)