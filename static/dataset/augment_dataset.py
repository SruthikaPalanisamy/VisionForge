import cv2
import numpy as np
import os
import random
import glob

def generate_3000_dataset(input_dir, output_dir, target_count=3000):
    images = glob.glob(os.path.join(input_dir, "*.jpg"))
    if not images:
        print("No images found!")
        return

    img_per_source = target_count // len(images)
    print(f"Generating {img_per_source} variants per image...")

    count = 0
    for img_path in images:
        img = cv2.imread(img_path)
        base_name = os.path.basename(img_path).split('.')[0]

        for i in range(img_per_source):
            # Random Rotation
            angle = random.uniform(-10, 10)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            aug_img = cv2.warpAffine(img, M, (w, h))

            # Random Brightness
            brightness = random.randint(-50, 50)
            aug_img = cv2.convertScaleAbs(aug_img, beta=brightness)

            # Random Noise
            if random.random() > 0.5:
                noise = np.random.normal(0, 15, aug_img.shape).astype(np.uint8)
                aug_img = cv2.add(aug_img, noise)

            save_path = os.path.join(output_dir, f"{base_name}_aug_{i}.jpg")
            cv2.imwrite(save_path, aug_img)
            count += 1
    
    print(f"Successfully created {count} images in {output_dir}")

# Usage
generate_3000_dataset("static/dataset/images", "static/dataset/augmented")
