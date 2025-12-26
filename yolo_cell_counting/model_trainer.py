import os
import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

class ModelTrainer:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.output_dir = data_processor.output_dir

    def train_yolo_model(self, epochs=50, imgsz=640, batch=4, model_type='yolov8n.pt'):
        """Train YOLO model for cell detection and classification"""
        print("Training YOLO model...")

        # Use the staining dataset YAML path
        data_yaml = os.path.join(self.output_dir, "yolo_staining_dataset", "data.yaml")

        if not os.path.exists(data_yaml):
            print("Data YAML not found. Creating dataset structure first...")
            # Load metadata and create staining dataset
            self.data_processor.load_metadata("metadata.csv")
            self.data_processor.prepare_staining_dataset()

        try:
            # Load a pretrained YOLO model
            model = YOLO(model_type)

            # Train the model
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch, #CHANGE BATCH IF CAN HANDLE
                name='cell_staining_classification',
                project=self.output_dir,
                exist_ok=True,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                patience=10,  # Early stopping patience, INCRESE TO 20 for MORE STABILITY
                lr0=0.01,    # Initial learning rate
                lrf=0.01,    # Final learning rate
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3.0,
                warmup_momentum=0.8,
                box=7.5,     # Box loss gain
                dfl=1.5,     # Distribution Focal Loss gain
                cls=5.0,     # Classification loss gain, INCREASE TO 10 for MORE AGRESSIVE CLASSIFICATION
                cos_lr=True,  # Use cosine learning rate schedule
            )

            print("Training completed!")

            # Save the best model explicitly
            best_model_path = self.get_best_model_path()
            if os.path.exists(best_model_path):
                # Copy best model to explicit location
                explicit_save_path = os.path.join(self.output_dir, 'trained_cell_staining_model.pt')
                import shutil
                shutil.copy2(best_model_path, explicit_save_path)
                print(f"Best model saved to: {explicit_save_path}")

            return model

        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def train_cell_type_model(self, epochs=50, imgsz=640, batch=8, model_type='yolov8n.pt'):
        """Alternative: Train YOLO model for cell type classification"""
        print("Training YOLO model for cell type classification...")

        # First prepare cell type dataset
        self.data_processor.load_metadata("metadata.csv")
        valid_cell_types = self.data_processor.prepare_cell_type_dataset()

        # Create custom data.yaml for cell types
        cell_type_yaml_path = self._create_cell_type_data_yaml(valid_cell_types)

        try:
            model = YOLO(model_type)

            results = model.train(
                data=cell_type_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                name='cell_type_classification',
                project=self.output_dir,
                exist_ok=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            print("Cell type training completed!")
            return model

        except Exception as e:
            print(f"Cell type training failed: {e}")
            return None

    def _create_cell_type_data_yaml(self, valid_cell_types):
        """Create data.yaml for cell type classification"""
        yolo_dir = os.path.join(self.output_dir, "yolo_celltype_dataset")
        os.makedirs(yolo_dir, exist_ok=True)

        data_yaml = {
            'path': os.path.abspath(yolo_dir),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(valid_cell_types),
            'names': valid_cell_types
        }

        yaml_path = os.path.join(yolo_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            import yaml
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"Created cell type data.yaml at: {yaml_path}")
        return yaml_path

    def predict_cell_count(self, image_path, model_path=None, conf_threshold=0.3, staining_type=None):
        """Predict cell count on an image with staining type information"""
        try:
            if model_path is None:
                model_path = self.get_best_model_path()
                if not os.path.exists(model_path):
                    print("No trained model found. Please train a model first.")
                    return 0, None, []

            model = YOLO(model_path)

            # Run prediction
            results = model.predict(
                source=image_path,
                conf=conf_threshold,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            # Process results
            result = results[0]
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Could not load image: {image_path}")
                return 0, None, []

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get class names from model
            class_names = model.names if hasattr(model, 'names') else {0: 'cell'}

            detection_info = []
            cell_counts_by_class = {}

            # Draw bounding boxes and collect detection info
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = class_names.get(class_id, f'class_{class_id}')

                    # Count cells by class
                    cell_counts_by_class[class_name] = cell_counts_by_class.get(class_name, 0) + 1

                    # Store detection info
                    detection_info.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    })

                    # Draw bounding box with class-specific color
                    color = self._get_class_color(class_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Label with class name and confidence
                    label = f'{class_name} {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            total_count = len(result.boxes) if result.boxes is not None else 0

            # Add count information text
            count_text = f'Total Cells: {total_count}'
            if staining_type:
                count_text += f' | Staining: {staining_type}'

            cv2.putText(frame, count_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Add class-wise counts
            y_offset = 80
            for class_name, count in cell_counts_by_class.items():
                class_text = f'{class_name}: {count}'
                cv2.putText(frame, class_text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                y_offset += 30

            print(f"Total Cells Detected: {total_count}")
            if cell_counts_by_class:
                print("Breakdown by class:")
                for class_name, count in cell_counts_by_class.items():
                    print(f"  {class_name}: {count}")

            # Display result
            plt.figure(figsize=(12, 8))
            plt.axis('off')
            plt.title(f'Cell Detection - {os.path.basename(image_path)}')
            plt.imshow(frame)
            plt.tight_layout()
            plt.show(block=False)

            return total_count, frame, detection_info

        except Exception as e:
            print(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return 0, None, []

    def _get_class_color(self, class_id):
        """Get consistent color for each class"""
        colors = [
            (255, 0, 0),    # Red - AF488
            (0, 255, 0),    # Green - DAPI
            (0, 0, 255),    # Blue - Cy3
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        return colors[class_id % len(colors)]

    def evaluate_model(self, model_path=None, data_yaml=None):
        """Evaluate model performance"""
        try:
            if model_path is None:
                model_path = self.get_best_model_path()

            if data_yaml is None:
                data_yaml = os.path.join(self.output_dir, "yolo_staining_dataset", "data.yaml")

            model = YOLO(model_path)

            # Run validation
            metrics = model.val(data=data_yaml)

            print("Model Evaluation Results:")
            print(f"mAP50: {metrics.box.map50:.4f}")
            print(f"mAP50-95: {metrics.box.map:.4f}")
            print(f"Precision: {metrics.box.precision:.4f}")
            print(f"Recall: {metrics.box.recall:.4f}")

            return metrics

        except Exception as e:
            print(f"Evaluation failed: {e}")
            return None

    def get_best_model_path(self):
        """Get path to best model from training"""
        staining_path = os.path.join(self.output_dir, 'cell_staining_classification', 'weights', 'best.pt')
        celltype_path = os.path.join(self.output_dir, 'cell_type_classification', 'weights', 'best.pt')

        if os.path.exists(staining_path):
            return staining_path
        elif os.path.exists(celltype_path):
            return celltype_path
        else:
            return os.path.join(self.output_dir, 'trained_cell_staining_model.pt')

    def get_explicit_model_path(self):
        """Get path to explicitly saved model"""
        return os.path.join(self.output_dir, 'trained_cell_staining_model.pt')