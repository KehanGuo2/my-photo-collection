import os
from pathlib import Path

categories = ['landscape', 'portrait', 'nature', 'urban']
base_dir = Path('photos')

for category in categories:
    folder = base_dir / category
    if not folder.exists():
        continue

    # Get all image files, sorted for consistency
    files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}])
    for idx, file in enumerate(files, 1):
        ext = file.suffix.lower()
        new_name = f"{category}-{idx:02d}{ext}"
        new_path = folder / new_name
        if file.name != new_name:
            print(f"Renaming {file.name} -> {new_name}")
            file.rename(new_path)
print("✅ All photos renamed!")