import sys

deps = {"cv2": "opencv-python", "numpy": "numpy", "pyzbar": "pyzbar"}
missing = []

print("Checking dependencies...")
for mod, pkg in deps.items():
    try:
        __import__(mod)
        print(f"  ✅ {mod}")
    except ImportError:
        print(f"  ❌ {mod} (pip install {pkg})")
        missing.append(pkg)

if not missing:
    print("\n🎉 All dependencies are installed and ready!")
else:
    print(f"\n⚠️  Install missing packages with: pip install {' '.join(missing)}")