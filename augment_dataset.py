"""
Data Augmentation Script for DeepFake Dataset
Generates augmented copies of images to increase dataset size
"""
import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import random

def apply_augmentation(image, aug_type):
    """Apply various augmentations to an image"""
    h, w = image.shape[:2]
    
    if aug_type == 'flip_h':
        # Horizontal flip
        return cv2.flip(image, 1)
    
    elif aug_type == 'flip_v':
        # Vertical flip
        return cv2.flip(image, 0)
    
    elif aug_type == 'rotate_15':
        # Rotate 15 degrees
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    elif aug_type == 'rotate_minus15':
        # Rotate -15 degrees
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, -15, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    elif aug_type == 'brightness_up':
        # Increase brightness
        return cv2.convertScaleAbs(image, alpha=1.0, beta=30)
    
    elif aug_type == 'brightness_down':
        # Decrease brightness
        return cv2.convertScaleAbs(image, alpha=1.0, beta=-30)
    
    elif aug_type == 'contrast_up':
        # Increase contrast
        return cv2.convertScaleAbs(image, alpha=1.3, beta=0)
    
    elif aug_type == 'contrast_down':
        # Decrease contrast
        return cv2.convertScaleAbs(image, alpha=0.7, beta=0)
    
    elif aug_type == 'blur':
        # Slight blur
        return cv2.GaussianBlur(image, (5, 5), 0.5)
    
    elif aug_type == 'noise':
        # Add noise
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        return cv2.add(image, noise)
    
    elif aug_type == 'shift_right':
        # Shift right
        matrix = np.float32([[1, 0, 20], [0, 1, 0]])
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    elif aug_type == 'shift_left':
        # Shift left
        matrix = np.float32([[1, 0, -20], [0, 1, 0]])
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    elif aug_type == 'zoom_in':
        # Zoom in slightly
        crop_h, crop_w = int(h * 0.9), int(w * 0.9)
        start_y, start_x = (h - crop_h) // 2, (w - crop_w) // 2
        cropped = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
        return cv2.resize(cropped, (w, h))
    
    elif aug_type == 'zoom_out':
        # Zoom out slightly
        scaled = cv2.resize(image, (int(w*0.9), int(h*0.9)))
        result = np.zeros_like(image)
        y_offset, x_offset = (h - scaled.shape[0]) // 2, (w - scaled.shape[1]) // 2
        result[y_offset:y_offset+scaled.shape[0], x_offset:x_offset+scaled.shape[1]] = scaled
        return result
    
    else:
        return image


def augment_image(input_path, output_dir, num_augmentations=5):
    """Generate augmented versions of a single image"""
    image = cv2.imread(input_path)
    if image is None:
        return 0
    
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    
    # List of augmentation types
    aug_types = [
        'flip_h', 'flip_v', 'rotate_15', 'rotate_minus15',
        'brightness_up', 'brightness_down', 'contrast_up', 'contrast_down',
        'blur', 'noise', 'shift_right', 'shift_left', 'zoom_in', 'zoom_out'
    ]
    
    # Randomly select augmentations
    selected_augs = random.sample(aug_types, min(num_augmentations, len(aug_types)))
    
    count = 0
    for i, aug_type in enumerate(selected_augs):
        try:
            aug_image = apply_augmentation(image, aug_type)
            output_path = os.path.join(output_dir, f"{name}_aug{i}_{aug_type}{ext}")
            cv2.imwrite(output_path, aug_image)
            count += 1
        except Exception as e:
            print(f"Error applying {aug_type} to {filename}: {e}")
    
    return count


def augment_dataset(data_dir, output_dir=None, num_augmentations=5):
    """
    Augment entire dataset (both Real and Fake folders)
    
    Args:
        data_dir: Path to dataset directory containing 'Real' and 'Fake' folders
        output_dir: Where to save augmented images (default: same as data_dir)
        num_augmentations: Number of augmented copies per image
    """
    if output_dir is None:
        output_dir = data_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_augmented = 0
    
    for class_name in ['Real', 'Fake']:
        input_class_dir = os.path.join(data_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        
        if not os.path.exists(input_class_dir):
            print(f"Warning: {input_class_dir} not found, skipping...")
            continue
        
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(input_class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"\nProcessing {class_name}: {len(image_files)} images")
        
        for filename in tqdm(image_files, desc=f"Augmenting {class_name}"):
            input_path = os.path.join(input_class_dir, filename)
            
            # Copy original
            original_output = os.path.join(output_class_dir, filename)
            if not os.path.exists(original_output):
                img = cv2.imread(input_path)
                cv2.imwrite(original_output, img)
            
            # Generate augmentations
            count = augment_image(input_path, output_class_dir, num_augmentations)
            total_augmented += count
    
    print(f"\n✓ Augmentation complete!")
    print(f"  Total augmented images created: {total_augmented}")
    
    # Print final counts
    for class_name in ['Real', 'Fake']:
        class_dir = os.path.join(output_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            print(f"  {class_name}: {count} images")


def main():
    parser = argparse.ArgumentParser(description='Augment DeepFake Dataset')
    parser.add_argument('--data-dir', type=str, required=True, 
                       help='Path to dataset directory (containing Real and Fake folders)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: same as data-dir)')
    parser.add_argument('--num-aug', type=int, default=5,
                       help='Number of augmentations per image (default: 5)')
    
    args = parser.parse_args()
    
    augment_dataset(args.data_dir, args.output_dir, args.num_aug)


if __name__ == '__main__':
    main()
