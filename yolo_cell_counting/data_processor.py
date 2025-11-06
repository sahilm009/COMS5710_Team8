import os
import yaml
import shutil
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DataProcessor:
    def __init__(self, data_root="dataset\\IDCIA", stainings=['AF488', 'DAPI', 'Cy3']):
        self.data_root = data_root
        self.stainings = stainings
        self.output_dir = "training_output"
        self.metadata = None
        self.bbox_size_pixels = 30  # Fixed 30px bounding boxes

        # Define augmentation strategies, number of augmentations per staining
        # NOTE, might need more Cy3 augmentations
        self.augmentation_plans = {
            'AF488': 20,
            'DAPI': 1,
            'Cy3': 6
        }

        # EXCLUDE ROTATIONS AND FLIPS THAT AFFECT COORDINATES, haven't figured it out yet
        # self.augmentation_transforms = A.Compose([
        #     A.HorizontalFlip(p=0.3),
        #     A.VerticalFlip(p=0.3),
        #     A.RandomRotate90(p=0.3),
        #     A.GaussianBlur(blur_limit=3, p=0.1),
        #     A.GaussNoise(var_limit=(5.0, 10.0), p=0.1),
        # ])
        self.augmentation_transforms = A.Compose([
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.MotionBlur(blur_limit=3, p=0.1),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        ])

    def load_metadata(self, metadata_path="dataset\\metadata.csv"):
        """Load and process metadata CSV file"""
        try:
            self.metadata = pd.read_csv(metadata_path)
            print(f"Loaded metadata with {len(self.metadata)} entries")
            print("Staining distribution:")
            staining_counts = self.metadata['staining'].value_counts()
            print(staining_counts)
            print("\nCell type distribution:")
            print(self.metadata['cell_type'].value_counts())
            return self.metadata
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return None

    def prepare_staining_dataset(self, force_refresh=False):
        """Prepare dataset for staining-based classification with data augmentation"""
        print("Preparing staining-based dataset with data augmentation...")

        if self.metadata is None:
            print("Please load metadata first using load_metadata()")
            return None

        # Create output directory structure
        yolo_dir = os.path.join(self.output_dir, "yolo_staining_dataset")
        completion_file = os.path.join(yolo_dir, "dataset_complete.txt")

        # Check if dataset already exists and is complete
        if not force_refresh and self._is_dataset_complete(yolo_dir, completion_file):
            print("✓ Dataset already exists and is complete. Skipping creation.")
            return yolo_dir

        # Create directories
        images_dir = os.path.join(yolo_dir, "images")
        labels_dir = os.path.join(yolo_dir, "labels")
        bbox_viz_dir = os.path.join(yolo_dir, "bbox_visualization")

        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(images_dir, split), exist_ok=True)
            os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
            os.makedirs(os.path.join(bbox_viz_dir, split), exist_ok=True)

        # Process each staining type
        staining_class_mapping = {staining: idx for idx, staining in enumerate(self.stainings)}
        print(f"Staining class mapping: {staining_class_mapping}")
        print(f"Using fixed bounding box size: {self.bbox_size_pixels}px")
        print(f"Augmentation plan: {self.augmentation_plans}")

        processed_count = 0
        missing_files = []
        augmentation_stats = {staining: 0 for staining in self.stainings}
        total_count = 0
        zero_cell_count_discarded = 0
        zero_cell_count_kept = 0

        for split in ['train', 'val', 'test']:
            print(f"\nProcessing {split} split...")

            for staining in self.stainings:
                staining_dir = os.path.join(self.data_root, f"{staining}_{split}")
                images_path = os.path.join(staining_dir, "images")
                ground_truth_path = os.path.join(staining_dir, "ground_truth")

                if not os.path.exists(images_path):
                    print(f"Warning: {images_path} not found")
                    continue

                # Get all TIFF files
                tiff_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.tif', '.tiff'))]
                print(f"Found {len(tiff_files)} original images for {staining}_{split}")

                for tiff_file in tqdm(tiff_files, desc=f"Processing {staining}_{split}"):
                    base_name = os.path.splitext(tiff_file)[0]

                    try:
                        # Convert image ID to match metadata
                        image_id = int(base_name)
                    except ValueError:
                        print(f"Warning: Could not convert {base_name} to integer ID")
                        continue

                    # Find this image in metadata
                    image_metadata = self.metadata[self.metadata['id'] == image_id]
                    if image_metadata.empty:
                        print(f"Warning: No metadata found for ID {image_id}")
                        missing_files.append(f"{staining}_{split}/{base_name}")
                        continue

                    # Process image
                    img_path = os.path.join(images_path, tiff_file)
                    processed_img = self._load_and_process_tiff(img_path)

                    if processed_img is not None:
                        # Process annotations
                        csv_path = os.path.join(ground_truth_path, f"{base_name}.csv")
                        if os.path.exists(csv_path):
                            df = pd.read_csv(csv_path)
                            total_count += 1

                            # Check if CSV has any valid rows (not just file existence)
                            if len(df) == 0 or ('X' in df.columns and 'Y' in df.columns and df[['X', 'Y']].notna().all(axis=1).sum() == 0):
                                # This is an empty image - apply downsampling
                                if np.random.random() > 0.1:  # Keep only 10%
                                    zero_cell_count_discarded += 1
                                    continue
                                else:
                                    zero_cell_count_kept += 1
                        if os.path.exists(csv_path):
                            # Determine how many augmented versions to create
                            num_augmentations = self.augmentation_plans[staining] if split == 'train' else 0

                            # Process original + augmented versions
                            for aug_idx in range(num_augmentations + 1):
                                if aug_idx == 0:
                                    # Original image
                                    aug_suffix = ""
                                    aug_image = processed_img
                                else:
                                    # Augmented version
                                    aug_suffix = f"_aug{aug_idx}"
                                    aug_image = self._apply_augmentation(processed_img)

                                # Create filenames
                                output_img_path = os.path.join(images_dir, split, f"{staining}_{base_name}{aug_suffix}.jpg")
                                label_path = os.path.join(labels_dir, split, f"{staining}_{base_name}{aug_suffix}.txt")
                                viz_path = os.path.join(bbox_viz_dir, split, f"{staining}_{base_name}{aug_suffix}.jpg")

                                # Skip if files already exist
                                if not force_refresh and os.path.exists(output_img_path) and os.path.exists(label_path):
                                    processed_count += 1
                                    augmentation_stats[staining] += 1
                                    continue

                                # Save augmented image
                                cv2.imwrite(output_img_path, aug_image)

                                # Create YOLO labels
                                self._create_yolo_labels(csv_path, label_path, aug_image.shape, staining_class_mapping[staining])

                                # Create visualization (only for original images to save space)
                                if aug_idx == 0:
                                    self._create_bbox_visualization(aug_image, csv_path, viz_path, staining_class_mapping[staining], staining)

                                processed_count += 1
                                augmentation_stats[staining] += 1
                        else:
                            print(f"Warning: No CSV ground truth found for {base_name}")
                            missing_files.append(f"{staining}_{split}/{base_name}.csv")
                    else:
                        print(f"Warning: Could not process image {img_path}")

        # Print augmentation statistics
        print(f"\n=== AUGMENTATION STATISTICS ===")
        print(f"Total images processed: {total_count} and kept zero-cell images: {zero_cell_count_kept}, discarded zero-cell images: {zero_cell_count_discarded}\n")
        for staining in self.stainings:
            original_count = len([f for f in os.listdir(os.path.join(self.data_root, f"{staining}_train", "images"))
                                if f.lower().endswith(('.tif', '.tiff'))])
            augmented_count = augmentation_stats[staining]
            print(f"{staining}: {original_count} → {augmented_count} images ({(augmented_count/original_count):.1f}x)")

        print(f"\nSuccessfully processed {processed_count} images")
        if missing_files:
            print(f"Missing files: {len(missing_files)}")
            for missing in missing_files[:10]:
                print(f"  - {missing}")

        # Create data.yaml and mark dataset as complete
        self._create_staining_data_yaml(yolo_dir, staining_class_mapping)
        self._mark_dataset_complete(yolo_dir, completion_file, processed_count)

        print(f"\n✓ Bounding box visualizations saved to: {bbox_viz_dir}")
        return yolo_dir

    def _apply_augmentation(self, image):
        """Apply augmentation to image (excluding brightness changes)"""
        try:
            # Convert BGR to RGB for albumentations
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply augmentation
            augmented = self.augmentation_transforms(image=image_rgb)
            augmented_image = augmented['image']

            # Convert back to BGR
            augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

            return augmented_image_bgr
        except Exception as e:
            print(f"Error applying augmentation: {e}")
            return image  # Return original if augmentation fails

    def _create_bbox_visualization(self, image, csv_path, output_path, class_id, staining_name):
        """Create visualization images with bounding boxes drawn"""
        try:
            # Create a copy of the image for visualization
            viz_image = image.copy()
            height, width = viz_image.shape[:2]

            # Read CSV coordinates
            df = pd.read_csv(csv_path)

            # Define colors for different stainings
            color_map = {
                'AF488': (0, 255, 0),    # Green
                'DAPI': (255, 0, 0),     # Blue
                'Cy3': (0, 165, 255)     # Orange
            }
            color = color_map.get(staining_name, (0, 255, 255))  # Yellow as default

            for _, row in df.iterrows():
                if 'X' in df.columns and 'Y' in df.columns:
                    x_center = row['X']
                    y_center = row['Y']

                    # Calculate bounding box coordinates in pixels
                    bbox_half = self.bbox_size_pixels // 2
                    x1 = int(x_center - bbox_half)
                    y1 = int(y_center - bbox_half)
                    x2 = int(x_center + bbox_half)
                    y2 = int(y_center + bbox_half)

                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width - 1, x2)
                    y2 = min(height - 1, y2)

                    # Draw bounding box
                    cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)

                    # Add staining label
                    label = f"{staining_name}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(viz_image, (x1, y1 - label_size[1] - 5),
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(viz_image, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Save visualization
            cv2.imwrite(output_path, viz_image)

        except Exception as e:
            print(f"Error creating bbox visualization {output_path}: {e}")

    def _is_dataset_complete(self, yolo_dir, completion_file):
        """Check if dataset is already complete"""
        if not os.path.exists(completion_file):
            return False

        # Check if all expected directories exist
        required_dirs = [
            os.path.join(yolo_dir, "images", "train"),
            os.path.join(yolo_dir, "images", "val"),
            os.path.join(yolo_dir, "images", "test"),
            os.path.join(yolo_dir, "labels", "train"),
            os.path.join(yolo_dir, "labels", "val"),
            os.path.join(yolo_dir, "labels", "test"),
            os.path.join(yolo_dir, "bbox_visualization", "train"),
            os.path.join(yolo_dir, "bbox_visualization", "val"),
            os.path.join(yolo_dir, "bbox_visualization", "test")
        ]

        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                return False

        # Check if data.yaml exists
        if not os.path.exists(os.path.join(yolo_dir, "data.yaml")):
            return False

        # Check if there are files in each directory
        for split in ['train', 'val', 'test']:
            images_split_dir = os.path.join(yolo_dir, "images", split)
            labels_split_dir = os.path.join(yolo_dir, "labels", split)
            viz_split_dir = os.path.join(yolo_dir, "bbox_visualization", split)

            if (not os.listdir(images_split_dir) or
                not os.listdir(labels_split_dir) or
                not os.listdir(viz_split_dir)):
                return False

        print("✓ Dataset validation passed - all files present")
        return True

    def _mark_dataset_complete(self, yolo_dir, completion_file, processed_count):
        """Mark dataset as complete by creating a completion file"""
        completion_info = {
            'created_at': pd.Timestamp.now().isoformat(),
            'processed_images': processed_count,
            'stainings': self.stainings,
            'data_root': self.data_root,
            'bbox_size_pixels': self.bbox_size_pixels,
            'augmentation_plan': self.augmentation_plans
        }

        with open(completion_file, 'w') as f:
            f.write(f"Dataset completed at: {completion_info['created_at']}\n")
            f.write(f"Processed images: {completion_info['processed_images']}\n")
            f.write(f"Stainings: {', '.join(completion_info['stainings'])}\n")
            f.write(f"Bounding box size: {completion_info['bbox_size_pixels']}px\n")
            f.write(f"Augmentation plan: {completion_info['augmentation_plan']}\n")

        print(f"Dataset marked as complete: {completion_file}")

    def _load_and_process_tiff(self, img_path):
        """Load and process TIFF image for YOLO training"""
        try:
            # Use PIL for TIFF handling
            pil_img = Image.open(img_path)
            img = np.array(pil_img)

            if img is None:
                print(f"Warning: Could not load TIFF image {img_path}")
                return None

            # Handle different bit depths
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            elif img.dtype == np.float32 or img.dtype == np.float64:
                if img.max() > 1.0:
                    img = (img / img.max()) * 255.0
                else:
                    img = img * 255.0
                img = img.astype(np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)

            # Convert to 3-channel if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                img = cv2.cvtColor(img[:,:,0], cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            return img

        except Exception as e:
            print(f"Error processing TIFF {img_path}: {e}")
            return None

    def _create_yolo_labels(self, csv_path, label_path, img_shape, class_id):
        """Create YOLO format labels with PROPER 30px bounding boxes"""
        try:
            df = pd.read_csv(csv_path)
            height, width = img_shape[:2]

            # Had ot change from width only to height as well because images are not square
            bbox_width_normalized = self.bbox_size_pixels / width
            bbox_height_normalized = self.bbox_size_pixels / height

            with open(label_path, 'w') as f:
                for _, row in df.iterrows():
                    if 'X' in df.columns and 'Y' in df.columns:
                        x_center = row['X'] / width
                        y_center = row['Y'] / height
                        # Use proper normalized dimensions
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width_normalized:.6f} {bbox_height_normalized:.6f}\n")

        except Exception as e:
            print(f"Error creating YOLO labels from {csv_path}: {e}")

    def _create_staining_data_yaml(self, yolo_dir, class_mapping):
        """Create data.yaml for staining-based classification"""
        data_yaml = {
            'path': os.path.abspath(yolo_dir),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_mapping),
            'names': list(class_mapping.keys())
        }

        yaml_path = os.path.join(yolo_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"Created data.yaml at: {yaml_path}")
        print(f"Classes: {list(class_mapping.keys())}")

    def preview_bboxes(self, num_samples=3):
        """Preview bounding boxes on sample images from each staining"""
        yolo_dir = os.path.join(self.output_dir, "yolo_staining_dataset")
        bbox_viz_dir = os.path.join(yolo_dir, "bbox_visualization")

        if not os.path.exists(bbox_viz_dir):
            print("No visualizations found. Please run prepare_staining_dataset() first.")
            return

        print(f"\nPreviewing {num_samples} samples from each split:")

        for split in ['train', 'val', 'test']:
            split_viz_dir = os.path.join(bbox_viz_dir, split)
            if os.path.exists(split_viz_dir):
                viz_files = [f for f in os.listdir(split_viz_dir) if f.endswith('.jpg')]
                print(f"\n{split.upper()} split - samples:")
                for viz_file in viz_files[:num_samples]:
                    print(f"  - {viz_file}")

    def prepare_cell_type_dataset(self):
        """Alternative: Prepare dataset for cell type classification"""
        if self.metadata is None:
            print("Please load metadata first using load_metadata()")
            return

        # Get unique cell types with sufficient samples
        cell_type_counts = self.metadata['cell_type'].value_counts()
        valid_cell_types = cell_type_counts[cell_type_counts >= 5].index.tolist()

        print(f"Training on {len(valid_cell_types)} cell types: {valid_cell_types}")

        return valid_cell_types

    def validate_dataset_quality(self):
        """Comprehensive dataset validation"""
        yolo_dir = os.path.join(self.output_dir, "yolo_staining_dataset")

        for split in ['train', 'val']:
            labels_dir = os.path.join(yolo_dir, "labels", split)
            images_dir = os.path.join(yolo_dir, "images", split)

            print(f"\n=== Validating {split} split ===")

            # Check label files
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            empty_labels = 0
            total_objects = 0

            for label_file in label_files[:50]:  # Sample first 50
                label_path = os.path.join(labels_dir, label_file)
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                if len(lines) == 0:
                    empty_labels += 1
                else:
                    total_objects += len(lines)

                # Validate coordinates
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x, y, w, h = map(float, parts)
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            print(f"Invalid coordinates in {label_file}: {x}, {y}, {w}, {h}")

            print(f"Empty label files: {empty_labels}/{len(label_files)}")
            print(f"Average objects per image: {total_objects/len(label_files):.2f}")

            # Check image-label correspondence
            image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
            print(f"Images: {len(image_files)}, Labels: {len(label_files)}")

    def debug_data_pipeline(self):
        """Debug the entire data processing pipeline"""
        yolo_dir = os.path.join(self.output_dir, "yolo_staining_dataset")

        # Check a few samples visually
        import matplotlib.pyplot as plt

        labels_dir = os.path.join(yolo_dir, "labels", "train")
        images_dir = os.path.join(yolo_dir, "images", "train")

        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')][:3]

        for label_file in label_files:
            image_file = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(images_dir, image_file)
            label_path = os.path.join(labels_dir, label_file)

            # Load and display
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            with open(label_path, 'r') as f:
                lines = f.readlines()

            h, w = img.shape[:2]

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(img)

            for line in lines:
                class_id, x_center, y_center, bbox_w, bbox_h = map(float, line.strip().split())

                # Convert to pixel coordinates
                x_center_px = x_center * w
                y_center_px = y_center * h
                bbox_w_px = bbox_w * w
                bbox_h_px = bbox_h * h

                # Calculate corners
                x1 = int(x_center_px - bbox_w_px/2)
                y1 = int(y_center_px - bbox_h_px/2)
                x2 = int(x_center_px + bbox_w_px/2)
                y2 = int(y_center_px + bbox_h_px/2)

                # Draw rectangle
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

            plt.title(f"Labels: {len(lines)} - File: {label_file}")
            plt.show(block=False)