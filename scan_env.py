# scan_env.py
import re
import os

patterns = [
    r'os\.environ\.get\(["\'](\w+)',
    r'os\.getenv\(["\'](\w+)',
    r'os\.environ\[["\'](\w+)',
]

found = {}
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in [".git", "venv", "__pycache__"]]
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            for pattern in patterns:
                for match in re.findall(pattern, content):
                    found.setdefault(match, []).append(path)

print(f"\n{'=' * 50}")
print(f"  {len(found)} variables d'environnement trouvées")
print(f"{'=' * 50}")
for var, files in sorted(found.items()):
    print(f"\n  {var}")
    for f in set(files):
        print(f"    └─ {f}")
