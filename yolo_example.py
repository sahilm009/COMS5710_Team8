import os
import yaml
import shutil
import random
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

class CellCountingYOLO:
    def __init__(self, data_root="dataset\\IDCIA", channels=['AF488', 'DAPI', 'Cy3']):
        self.data_root = data_root
        self.channels = channels
        self.output_dir = "yolo_cell_counting"

    def create_composite_images(self):
        """Create composite images from multiple channels for YOLO training"""
        print("Creating composite images from channels...")

        composite_dir = os.path.join(self.output_dir, "composite_images")
        os.makedirs(composite_dir, exist_ok=True)

        # Process each split (train, test, val)
        for split in ['train', 'test', 'val']:
            split_dir = os.path.join(composite_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            # Get base filenames from first channel
            first_channel_dir = os.path.join(self.data_root, f"{self.channels[0]}_{split}")
            if not os.path.exists(first_channel_dir):
                print(f"Warning: {first_channel_dir} not found, skipping...")
                continue

            # Get all image files
            image_files = [f for f in os.listdir(first_channel_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

            print(f"Processing {len(image_files)} images in {split} set...")

            # Use tqdm for progress bar
            for filename in tqdm(image_files, desc=f"Processing {split} images"):
                base_name = os.path.splitext(filename)[0]

                # Load and combine channels
                channel_images = []
                for channel in self.channels:
                    channel_dir = os.path.join(self.data_root, f"{channel}_{split}")
                    img_path = self._find_image_file(channel_dir, base_name)

                    if img_path:
                        img = self._load_image(img_path)
                        if img is not None:
                            channel_images.append(img)
                        else:
                            print(f"Warning: Could not load {base_name} from {channel_dir}")
                            break
                    else:
                        print(f"Warning: {base_name} not found in {channel_dir}")
                        break

                if len(channel_images) == len(self.channels):
                    # Create composite image
                    composite = self._create_composite(channel_images)

                    if composite is not None:
                        # Ensure uint8
                        if composite.dtype != np.uint8:
                            composite = cv2.normalize(composite, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                        # Ensure RGB (3 channels)
                        if len(composite.shape) == 2:
                            composite = cv2.cvtColor(composite, cv2.COLOR_GRAY2RGB)
                        elif composite.shape[2] == 1:
                            composite = cv2.cvtColor(composite, cv2.COLOR_GRAY2RGB)

                        output_path = os.path.join(split_dir, f"{base_name}.jpg")
                        cv2.imwrite(output_path, composite)
                    else:
                        print(f"Warning: Failed to create composite for {base_name}")

    def _find_image_file(self, directory, base_name):
        """Find image file with various extensions"""
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.TIF', '.TIFF']:
            img_path = os.path.join(directory, base_name + ext)
            if os.path.exists(img_path):
                return img_path
        return None

    def _load_image(self, img_path):
        """Load image and convert to 8-bit RGB with proper type handling"""
        try:
            # Try different loading methods
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                # Try with PIL as fallback
                pil_img = Image.open(img_path)
                img = np.array(pil_img)

            if img is None:
                print(f"Warning: Could not load image {img_path}")
                return None

            # Convert to float first to avoid type promotion issues
            img = img.astype(np.float32)

            # Handle different original data types by normalizing to 0-255 range
            if img.max() > 255:
                # Likely 16-bit or higher
                img = (img / img.max()) * 255.0
            elif img.max() <= 1.0 and img.max() > 0:
                # Float image in 0-1 range
                img = img * 255.0

            # Now convert to uint8
            img = img.astype(np.uint8)

            # Ensure 3 channels
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            return img

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

    def _create_composite(self, channel_images):
        """Create composite image with proper type handling"""
        try:
            # Ensure all images have the same shape
            target_shape = channel_images[0].shape
            processed_channels = []

            for img in channel_images:
                # Resize if necessary to match first image
                if img.shape != target_shape:
                    img = cv2.resize(img, (target_shape[1], target_shape[0]))

                # Convert to float32 for safe averaging
                img_float = img.astype(np.float32)
                processed_channels.append(img_float)

            # Stack and average in float space
            stacked = np.stack(processed_channels, axis=0)
            composite = np.mean(stacked, axis=0)

            # Convert back to uint8
            composite = np.clip(composite, 0, 255).astype(np.uint8)

            return composite

        except Exception as e:
            print(f"Error creating composite: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_yolo_dataset_structure(self):
        """Create proper YOLO dataset structure with data.yaml"""
        print("Creating YOLO dataset structure...")

        composite_dir = os.path.join(self.output_dir, "composite_images")
        yolo_dir = os.path.join(self.output_dir, "yolo_dataset")

        # Create YOLO directory structure
        images_dir = os.path.join(yolo_dir, "images")
        labels_dir = os.path.join(yolo_dir, "labels")

        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(images_dir, split), exist_ok=True)
            os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

        # Copy images to YOLO structure
        all_images = []
        for split in ['train', 'val', 'test']:
            source_split_dir = os.path.join(composite_dir, split)
            if not os.path.exists(source_split_dir):
                continue

            for img_file in os.listdir(source_split_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src_path = os.path.join(source_split_dir, img_file)
                    dst_path = os.path.join(images_dir, split, img_file)
                    shutil.copy2(src_path, dst_path)
                    all_images.append((split, img_file))

        # Create dummy annotations (REPLACE WITH YOUR REAL ANNOTATIONS)
        self._create_dummy_annotations(labels_dir, all_images)

        # Create data.yaml file
        self._create_data_yaml(yolo_dir)

        print(f"YOLO dataset created at: {yolo_dir}")
        return yolo_dir

    def _create_dummy_annotations(self, labels_dir, all_images):
        """Create dummy YOLO format annotations for demonstration"""
        print("Creating dummy annotations (replace with your real annotations)...")

        for split, img_file in all_images:
            # Get image dimensions
            img_path = os.path.join(self.output_dir, "yolo_dataset", "images", split, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            height, width = img.shape[:2]

            # Create annotation file path
            base_name = os.path.splitext(img_file)[0]
            label_path = os.path.join(labels_dir, split, base_name + ".txt")

            # Generate dummy bounding boxes (cells)
            num_cells = random.randint(5, 30)
            with open(label_path, 'w') as f:
                for _ in range(num_cells):
                    # YOLO format: class x_center y_center width height (normalized 0-1)
                    class_id = 0  # Assuming only one class "cell"

                    # Random bounding box
                    bbox_width = random.uniform(0.02, 0.1)  # 2-10% of image width
                    bbox_height = random.uniform(0.02, 0.1)  # 2-10% of image height

                    x_center = random.uniform(bbox_width/2, 1 - bbox_width/2)
                    y_center = random.uniform(bbox_height/2, 1 - bbox_height/2)

                    # Write to file
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

    def _create_data_yaml(self, yolo_dir):
        """Create data.yaml configuration file for YOLO"""
        data_yaml = {
            'path': os.path.abspath(yolo_dir),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,  # number of classes
            'names': ['cell']  # class names
        }

        yaml_path = os.path.join(yolo_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"Created data.yaml at: {yaml_path}")
        return yaml_path

    def train_yolo_model(self, epochs=50, imgsz=640, batch=8):
        """Train YOLO model for cell detection and save it"""
        print("Training YOLO model...")

        data_yaml = os.path.join(self.output_dir, "yolo_dataset", "data.yaml")

        if not os.path.exists(data_yaml):
            print("Data YAML not found. Creating dataset structure first...")
            self.create_yolo_dataset_structure()

        try:
            # Load a pretrained YOLO model
            model = YOLO('yolov8n.pt')

            # Train the model
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                name='cell_counting',
                project=self.output_dir,
                exist_ok=True,
                device='cuda'  # Force GPU usage
            )

            print("Training completed!")

            # Explicitly save the model
            model_save_path = os.path.join(self.output_dir, 'trained_cell_model.pt')
            model.save(model_save_path)
            print(f"Model explicitly saved to: {model_save_path}")

            return model

        except Exception as e:
            print(f"Training failed: {e}")
            return None

    def predict_cell_count(self, image_path, model_path=None, conf_threshold=0.3):
        """Predict cell count on an image"""
        try:
            if model_path is None:
                # Try to find the best trained model
                model_path = os.path.join(self.output_dir, 'cell_counting', 'weights', 'best.pt')
                if not os.path.exists(model_path):
                    print("No trained model found. Using pretrained YOLO for demonstration.")
                    model = YOLO('yolov8n.pt')
                else:
                    model = YOLO(model_path)
            else:
                model = YOLO(model_path)

            # Run prediction
            results = model.predict(source=image_path, conf=conf_threshold, device='cuda')

            # Process results
            result = results[0]
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Could not load image: {image_path}")
                return 0, None

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Draw bounding boxes
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            count = len(result.boxes) if result.boxes is not None else 0

            # Add count text
            cv2.putText(frame, f'Cells: {count}', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

            print(f"Total Cells Detected: {count}")

            # Display result
            plt.figure(figsize=(12, 8))
            plt.axis('off')
            plt.imshow(frame)
            plt.show()

            return count, frame

        except Exception as e:
            print(f"Prediction failed: {e}")
            return 0, None

    def save_trained_model(self, model, model_name="cell_counting_model.pt"):
        """Explicitly save the trained model"""
        model_save_path = os.path.join(self.output_dir, model_name)

        # Save the model
        if hasattr(model, 'save'):
            model.save(model_save_path)
        else:
            torch.save(model.state_dict(), model_save_path)

        print(f"Model saved to: {model_save_path}")
        return model_save_path

    def csv_to_yolo_txt(self, csv_dir, image_dir, labels_dir, default_bbox_size=0.02):
        """
        Convert CSV cell center coordinates to YOLO label .txt files.

        Args:
            csv_dir (str): Path to directory containing CSV files (one per image).
            image_dir (str): Path to directory containing corresponding images.
            labels_dir (str): Output directory for YOLO .txt label files.
            default_bbox_size (float): Default bounding box width/height (as fraction of image size).
        """
        os.makedirs(labels_dir, exist_ok=True)
        csv_files = [f for f in os.listdir(csv_dir) if f.lower().endswith('.csv')]

        print(f"Converting {len(csv_files)} CSV files to YOLO .txt format...")

        for csv_file in tqdm(csv_files, desc="CSV→YOLO"):
            base_name = os.path.splitext(csv_file)[0]

            # Read CSV file (expects columns: x, y)
            csv_path = os.path.join(csv_dir, csv_file)
            try:
                df = pd.read_csv(csv_path)
                if 'X' not in df.columns or 'Y' not in df.columns:
                    print(f"Skipping {csv_file}: missing 'x' or 'y' columns.")
                    continue
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue

            # Get image size
            img_path = self._find_image_file(image_dir, base_name)
            if not img_path:
                print(f"Image not found for {base_name}, skipping...")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image {img_path}, skipping...")
                continue

            height, width = img.shape[:2]

            # Output .txt label path
            label_path = os.path.join(labels_dir, base_name + ".txt")

            # Write YOLO annotations
            with open(label_path, 'w') as f:
                for _, row in df.iterrows():
                    x_center = row['X'] / width
                    y_center = row['Y'] / height

                    # YOLO format: class x_center y_center width height
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {default_bbox_size:.6f} {default_bbox_size:.6f}\n")

        print(f"YOLO labels saved to: {labels_dir}")

    def evaluate_model(self, test_images_dir, test_labels_dir, model_path=None, conf_threshold=0.3):
        """
        Evaluate YOLO model performance on test dataset without displaying images

        Args:
            test_images_dir (str): Directory containing test images
            test_labels_dir (str): Directory containing ground truth YOLO label files
            model_path (str): Path to trained model weights
            conf_threshold (float): Confidence threshold for detection

        Returns:
            dict: Dictionary containing evaluation metrics
        """

        print("Evaluating model performance...")

        # Load model
        if model_path is None:
            model_path = os.path.join(self.output_dir, 'cell_counting', 'weights', 'best.pt')

        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return None

        model = YOLO(model_path)

        # Get test images
        test_images = [f for f in os.listdir(test_images_dir)
                       if f.lower().endswith(('.tiff', '.jpg', '.jpeg'))]

        if not test_images:
            print("No test images found!")
            return None

        # Initialize lists for metrics
        predicted_counts = []
        true_counts = []
        absolute_errors = []
        squared_errors = []
        within_5_percent = []

        print(f"Processing {len(test_images)} test images...")

        for img_file in tqdm(test_images, desc="Evaluating"):
            base_name = os.path.splitext(img_file)[0]

            # Get ground truth count from label file
            label_file = os.path.join(test_labels_dir, base_name + ".txt")
            true_count = 0
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    true_count = len(f.readlines())

            # Get predicted count without displaying image
            img_path = os.path.join(test_images_dir, img_file)
            try:
                # Run prediction silently
                results = model.predict(source=img_path, conf=conf_threshold, verbose=False)
                predicted_count = len(results[0].boxes) if results[0].boxes is not None else 0

                # Store results
                predicted_counts.append(predicted_count)
                true_counts.append(true_count)

                # Calculate errors
                if true_count > 0:  # Avoid division by zero
                    abs_error = abs(predicted_count - true_count)
                    absolute_errors.append(abs_error)
                    squared_errors.append(abs_error ** 2)

                    # Check if within 5% of true count (ACP metric)
                    if abs_error <= 0.05 * true_count:
                        within_5_percent.append(1)
                    else:
                        within_5_percent.append(0)
                else:
                    # Handle case where true_count is 0
                    absolute_errors.append(abs(predicted_count))
                    squared_errors.append(predicted_count ** 2)
                    within_5_percent.append(1 if predicted_count == 0 else 0)

            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue

        # Calculate metrics
        metrics = {}

        # Mean Absolute Error (MAE)
        metrics['mae'] = np.mean(absolute_errors) if absolute_errors else 0

        # Root Mean Squared Error (RMSE)
        metrics['rmse'] = np.sqrt(np.mean(squared_errors)) if squared_errors else 0

        # Accuracy within 5% (ACP)
        metrics['acp'] = (np.mean(within_5_percent) * 100) if within_5_percent else 0

        # Overall Accuracy (exact matches)
        exact_matches = np.sum(np.array(predicted_counts) == np.array(true_counts))
        metrics['accuracy'] = (exact_matches / len(true_counts)) * 100 if true_counts else 0

        # Additional metrics
        metrics['total_images'] = len(test_images)
        metrics['total_cells_true'] = np.sum(true_counts)
        metrics['total_cells_predicted'] = np.sum(predicted_counts)

        # Correlation coefficient
        if len(predicted_counts) > 1:
            correlation = np.corrcoef(predicted_counts, true_counts)[0, 1]
            metrics['correlation'] = correlation
        else:
            metrics['correlation'] = 0

        # Mean Absolute Percentage Error (MAPE)
        non_zero_mask = np.array(true_counts) > 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((np.array(predicted_counts)[non_zero_mask] - np.array(true_counts)[non_zero_mask]) / np.array(true_counts)[non_zero_mask])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = 0

        # Print comprehensive results
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Total Images Tested: {metrics['total_images']}")
        print(f"Total True Cells: {metrics['total_cells_true']}")
        print(f"Total Predicted Cells: {metrics['total_cells_predicted']}")
        print(f"Detection Ratio: {metrics['total_cells_predicted']/max(1, metrics['total_cells_true']):.3f}")
        print("-" * 60)
        print(f"Accuracy (exact match): {metrics['accuracy']:.2f}%")
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.3f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.3f}")
        print(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
        print(f"Accuracy within 5% (ACP): {metrics['acp']:.2f}%")
        print(f"Correlation (r): {metrics['correlation']:.3f}")
        print("="*60)

        # Print sample comparisons
        print("\nSAMPLE PREDICTIONS (first 10 images):")
        print("-" * 55)
        print(f"{'Image':<15} {'True':<6} {'Predicted':<10} {'Error':<8} {'Within 5%':<12}")
        print("-" * 55)

        for i in range(min(10, len(test_images))):
            img_name = test_images[i][:12] + "..." if len(test_images[i]) > 12 else test_images[i]
            error = predicted_counts[i] - true_counts[i]
            within_5p = "Yes" if within_5_percent[i] else "No"
            print(f"{img_name:<15} {true_counts[i]:<6} {predicted_counts[i]:<10} {error:<8} {within_5p:<12}")

        # Print summary statistics
        print(f"\nSUMMARY STATISTICS:")
        print(f"Min True Count: {np.min(true_counts) if true_counts else 0}")
        print(f"Max True Count: {np.max(true_counts) if true_counts else 0}")
        print(f"Mean True Count: {np.mean(true_counts) if true_counts else 0:.2f}")
        print(f"Min Predicted Count: {np.min(predicted_counts) if predicted_counts else 0}")
        print(f"Max Predicted Count: {np.max(predicted_counts) if predicted_counts else 0}")
        print(f"Mean Predicted Count: {np.mean(predicted_counts) if predicted_counts else 0:.2f}")

        return metrics

    def quick_evaluate(self, model_path=None, conf_threshold=0.3):
        """
        Quick evaluation using the test set from the YOLO dataset structure
        """
        test_images_dir = os.path.join(self.output_dir, "yolo_dataset", "images", "test")
        test_labels_dir = os.path.join(self.output_dir, "yolo_dataset", "labels", "test")

        if not os.path.exists(test_images_dir):
            print(f"Test images directory not found: {test_images_dir}")
            return None

        return self.evaluate_model(test_images_dir, test_labels_dir, model_path, conf_threshold)


def main():
    # Initialize cell counting
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    test = True
    cell_counter = CellCountingYOLO(data_root="dataset\\IDCIA", channels=['AF488', 'DAPI', 'Cy3'])

    if (test):
        metrics = cell_counter.evaluate_model(
            test_images_dir="yolo_cell_counting/yolo_dataset/images/test",
            test_labels_dir="yolo_cell_counting/yolo_dataset/labels/test",
            #model_path="yolo_cell_counting/yolo_dataset/best.pt",
            model_path="yolo_cell_counting/trained_cell_model.pt",
            conf_threshold=0.3
        )

        # Access individual metrics
        if metrics:
            print(f"ACP: {metrics['acp']:.2f}%")
            print(f"MAE: {metrics['mae']:.3f}")
            print(f"RMSE: {metrics['rmse']:.3f}")
            print(f"Accuracy: {metrics['accuracy']:.2f}%")

        print("Finished testing.")
        return


    # Uncomment and modify these paths if you have CSV annotations:
    # cell_counter.csv_to_yolo_txt(
    #     csv_dir="dataset/IDCIA/DAPI_val/ground_truth",
    #     image_dir="dataset/IDCIA/DAPI_val/images",
    #     labels_dir="yolo_cell_counting/yolo_dataset/labels/val"
    # )

    # Step 1: Create composite images
    print("Step 1: Creating composite images...")
    cell_counter.create_composite_images()

    # Step 2: Create YOLO dataset structure
    print("\nStep 2: Creating YOLO dataset structure...")
    cell_counter.create_yolo_dataset_structure()

    root = "yolo_cell_counting/yolo_dataset/images"

    #check uint8
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(root, split)
        if not os.path.exists(split_dir):
            continue
        for f in os.listdir(split_dir):
            path = os.path.join(split_dir, f)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.dtype == np.uint16:
                # Convert 16-bit → 8-bit by normalizing intensity range
                img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(path, img_8bit)
                print(f"Converted {path} to uint8")

    for split in ['train', 'val', 'test']:
        for f in os.listdir(os.path.join(root, split)):
            img = cv2.imread(os.path.join(root, split, f), cv2.IMREAD_UNCHANGED)
            assert img.dtype == np.uint8, f"{f} still {img.dtype}"
        print("✅ All images are now uint8")

    #check rgb channels = 3
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(root, split)
        if not os.path.exists(split_dir):
            continue
        for f in os.listdir(split_dir):
            path = os.path.join(split_dir, f)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            # If grayscale (2D), convert to RGB
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                cv2.imwrite(path, img)
                print(f"Converted {path} from grayscale to RGB (3 channels)")

            # If single channel (3D but channel=1), convert to RGB
            elif len(img.shape) == 3 and img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                cv2.imwrite(path, img)
                print(f"Converted {path} from single-channel to RGB (3 channels)")

            # If 4 channels (RGBA), convert to RGB
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # or COLOR_RGBA2RGB
                cv2.imwrite(path, img)
                print(f"Converted {path} from RGBA to RGB (4 -> 3 channels)")

            # If already 3 channels, do nothing
            elif len(img.shape) == 3 and img.shape[2] == 3:
                continue

            # Handle any other unexpected cases
            else:
                print(f"Unexpected image format at {path}: shape {img.shape}")


    for split in ['train', 'val', 'test']:
        for f in os.listdir(os.path.join(root, split)):
            img = cv2.imread(os.path.join(root, split, f))
            if img is not None and img.shape[2] != 3:
                print(f"{f} has {img.shape[2]} channels")

    # Step 3: Train the model (optional - comment out if you just want to test)
    print("\nStep 3: Training YOLO model...")
    trained_model = cell_counter.train_yolo_model(epochs=10, imgsz=640, batch=4)  # Reduced for testing

    # Verify model is saved
    if trained_model is not None:
        # Check for automatically saved best model
        best_model_path = os.path.join(cell_counter.output_dir, 'cell_counting', 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            print(f"✓ Best model automatically saved at: {best_model_path}")

        # Check for explicitly saved model
        explicit_model_path = os.path.join(cell_counter.output_dir, 'trained_cell_model.pt')
        if os.path.exists(explicit_model_path):
            print(f"✓ Model explicitly saved at: {explicit_model_path}")

        # Test loading the model
        try:
            loaded_model = YOLO(best_model_path)
            print("✓ Model can be successfully loaded for inference")
        except Exception as e:
            print(f"✗ Error loading model: {e}")

    print("Finished training and saving.")

if __name__ == "__main__":
    main()