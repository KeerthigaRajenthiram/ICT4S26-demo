import os

# 1. Project Root
# We anchor paths to the location of THIS file (.../automl_kb/config.py)
# This prevents errors when running scripts from different directories.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 2. Data Directory
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 3. Dependent Paths
DB_PATH = os.path.join(DATA_DIR, "runs.db")
ARTIFACT_DIR = os.path.join(DATA_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

OPENML_CACHE_DIR = os.path.join(DATA_DIR, "openml_cache")
os.makedirs(OPENML_CACHE_DIR, exist_ok=True)

# 4. Framework Settings
H2O_MAX_MEM = "2G"

# Java Path (for H2O)
# Checks environment variable first, then defaults to local path
JAVA_HOME_CUSTOM = os.environ.get(
    "JAVA_HOME_CUSTOM", 
    os.path.expanduser("~/java_home/jdk-17.0.2/bin")
)

# 5. Toggles
ENABLE_ENERGY_TRACKING = True