import getpass
import hashlib
import os
import sys
import shutil
import subprocess
from datetime import datetime

try:
    import pytest
    pytest_version = pytest.__version__
except ImportError:
    pytest_version = None


def get_quarto_version():
    quarto_path = shutil.which("quarto")
    if quarto_path is None:
        return None
    try:
        result = subprocess.run(
            ["quarto", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "<error>"


username = getpass.getuser()
timestamp = datetime.now().isoformat(timespec="seconds")

conda_env = os.environ.get("CONDA_DEFAULT_ENV", "<not set>")
python_executable = sys.executable
python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
quarto_version = get_quarto_version()

checks = {
    "Environment name": conda_env == "pycw",
    "Python version": python_version == "3.12",
    "pytest installed": pytest_version is not None,
    "Quarto available": quarto_version not in (None, "<error>"),
}

token = hashlib.sha256(
    f"{username}|{timestamp}|{python_executable}".encode()
).hexdigest()[:8]

print("=== PYINTRO ENVIRONMENT CHECK ===")
print()
print(f"Timestamp: {timestamp}")
print(f"Username: {username}")
print(f"Verification code: {token}")
print()
print(f"Environment : {conda_env}")
print(f"Python      : {python_version}")
print(f"Executable  : {python_executable}")
print(f"pytest      : {pytest_version}")
print(f"Quarto      : {quarto_version or 'Not found'}")
print()

all_passed = True

for name, passed in checks.items():
    print(f"{'✓' if passed else '✗'} {name}")
    all_passed = all_passed and passed

print()

if all_passed:
    print("STATUS: OK")
    sys.exit(0)
else:
    print("STATUS: FAILED")
    sys.exit(1)
