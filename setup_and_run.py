"""One-command launcher: installs what's missing, rebuilds the UI if needed, starts the web app.

    python setup_and_run.py            # install + (build UI if Node exists) + start server
    python setup_and_run.py --demo     # just run the synthetic demo
"""
import importlib.util
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")


def have(mod):
    return importlib.util.find_spec(mod) is not None


def sh(cmd, cwd=ROOT):
    print(">", " ".join(cmd))
    return subprocess.call(cmd, cwd=cwd, shell=(os.name == "nt" and cmd[0] in ("npm", "npx")))


def main():
    need = {"numpy": "numpy", "scipy": "scipy", "cv2": "opencv-python", "skimage": "scikit-image",
            "matplotlib": "matplotlib", "fastapi": "fastapi", "uvicorn": "uvicorn",
            "multipart": "python-multipart", "httpx": "httpx", "pytest": "pytest"}
    missing = [pkg for mod, pkg in need.items() if not have(mod)]
    if missing:
        sh([sys.executable, "-m", "pip", "install", *missing])

    if "--demo" in sys.argv:
        sys.exit(sh([sys.executable, "run.py", "--demo"]))

    # Rebuild UI so the segmentation controls exist (prebuilt dist is older). Optional: needs Node.
    marker = os.path.join(WEB, "dist", ".built_v3")
    if shutil.which("npm") and not os.path.exists(marker):
        if sh(["npm", "install"], WEB) == 0 and sh(["npm", "run", "build"], WEB) == 0:
            open(marker, "w").write("ok")
        else:
            print("UI rebuild failed; using the prebuilt UI (segmentation controls may be missing).")
    elif not shutil.which("npm"):
        print("Node not found: using the prebuilt UI (install Node 18+ to get segmentation controls).")

    sys.exit(sh([sys.executable, "serve.py"]))


if __name__ == "__main__":
    main()
