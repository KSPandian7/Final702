import os
import shutil

sketch_src = r"E:\19AI702-P1\dataset\sketches"
photo_src  = r"E:\19AI702-P1\dataset\photos"

sketch_dst = r"E:\19AI702-P1\data\sketches"
photo_dst  = r"E:\19AI702-P1\data\photos"

os.makedirs(sketch_dst, exist_ok=True)
os.makedirs(photo_dst, exist_ok=True)

sketch_files = sorted(os.listdir(sketch_src))
photo_files = sorted(os.listdir(photo_src))

for i, (s, p) in enumerate(zip(sketch_files, photo_files)):
    shutil.copy(
        os.path.join(sketch_src, s),
        os.path.join(sketch_dst, f"person_{i:03}.jpg")
    )
    shutil.copy(
        os.path.join(photo_src, p),
        os.path.join(photo_dst, f"person_{i:03}.jpg")
    )

print("✅ Dataset prepared successfully!")
