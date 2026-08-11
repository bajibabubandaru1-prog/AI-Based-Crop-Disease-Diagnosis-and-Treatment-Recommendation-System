import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "plantvillage"
    / "archive"
    / "PlantVillage"
    / "val"
)

random.seed(42)

class_folders = sorted(
    folder for folder in VAL_DIR.iterdir() if folder.is_dir()
)

columns = 5
rows = math.ceil(len(class_folders) / columns)

fig, axes = plt.subplots(rows, columns, figsize=(18, 24))
axes = axes.flatten()

for axis, class_folder in zip(axes, class_folders):
    images = [
        path
        for path in class_folder.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if not images:
        axis.set_title(f"{class_folder.name}\nNo images")
        axis.axis("off")
        continue

    selected_image = random.choice(images)
    image = Image.open(selected_image).convert("RGB")

    axis.imshow(image)
    axis.set_title(
        class_folder.name.replace("___", "\n"),
        fontsize=8,
    )
    axis.axis("off")

for axis in axes[len(class_folders):]:
    axis.axis("off")

plt.suptitle(
    f"PlantVillage Dataset — One Sample from Each of {len(class_folders)} Classes",
    fontsize=18,
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

output_path = PROJECT_ROOT / "class_samples.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"Displayed classes: {len(class_folders)}")
print(f"Saved image grid to: {output_path}")

plt.show()