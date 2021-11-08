from pathlib import Path
import os

PACKAGE_PATH = Path(__file__).resolve().parent
DATA_PATH = PACKAGE_PATH /'dataset' / 'data'

CACHE_PATH = Path(os.environ.get('TEMPORIS_CACHE_PATH', Path.home() / '.temporis' / 'cache'))
CACHE_PATH.mkdir(parents=True, exist_ok=True)


__version__ = 0.5
