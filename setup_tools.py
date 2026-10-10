"""Download the Real-ESRGAN ncnn-Vulkan engine (works on AMD / NVIDIA / Intel GPUs).

Run once:  py -3.11 setup_tools.py
"""
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools" / "realesrgan"
URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
       "realesrgan-ncnn-vulkan-20220424-windows.zip")


def main() -> None:
    exe = TOOLS / "realesrgan-ncnn-vulkan.exe"
    if exe.exists():
        print(f"Already installed: {exe}")
        return
    dl_dir = ROOT / "tools" / "_download"
    dl_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dl_dir / "realesrgan.zip"
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, zip_path)
    extract_dir = dl_dir / "extract"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    found = next(extract_dir.rglob("realesrgan-ncnn-vulkan.exe"), None)
    if not found:
        sys.exit("realesrgan-ncnn-vulkan.exe not found in archive")
    TOOLS.parent.mkdir(parents=True, exist_ok=True)
    if TOOLS.exists():
        shutil.rmtree(TOOLS)
    shutil.copytree(found.parent, TOOLS)
    shutil.rmtree(dl_dir)
    print(f"Installed to {TOOLS}")
    for m in sorted((TOOLS / "models").glob("*.param")):
        print("  model:", m.stem)


if __name__ == "__main__":
    main()
