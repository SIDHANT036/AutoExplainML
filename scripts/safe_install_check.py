import subprocess
import sys
import tempfile
import os

PACKAGE_NAME = "autoexplainml"

def run(cmd):
    print("\n>>>", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise Exception("FAILED COMMAND")

def test_clean_install():
    print("🔍 Creating clean environment...")

    venv_path = tempfile.mkdtemp()
    python = sys.executable

    # create venv
    run([python, "-m", "venv", venv_path])

    pip = os.path.join(venv_path, "bin", "pip")
    py = os.path.join(venv_path, "bin", "python")

    # upgrade pip
    run([pip, "install", "--upgrade", "pip"])

    # install your package
    run([pip, "install", PACKAGE_NAME])

    # test import
    run([py, "-c", "import autoexplainml; print('IMPORT OK')"])

    print("\n✅ SAFE INSTALL PASSED")

if __name__ == "__main__":
    test_clean_install()