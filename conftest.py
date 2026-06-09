import sys
from pathlib import Path

# Make the repo root importable (so `import models` works) when running pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent))
