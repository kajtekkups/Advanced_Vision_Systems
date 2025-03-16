from pathlib import Path
import cv2

#import files path
parent_dir = Path(__file__).resolve().parent.parent
images_dir = parent_dir / "zaw_avs_materials" / "lab02_cfd" / "pedestrian" / "input"

for i in range(300, 1100, 1) :
    image_path = images_dir / f'in{i:06d}.jpg'
    I = cv2.imread(image_path)
    cv2.imshow("I", I)
    cv2.waitKey()

