import torch
import os
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator

def main():
    # Initialize cell counting
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    data_processor = DataProcessor(data_root="dataset\\IDCIA", stainings=['AF488', 'DAPI', 'Cy3'])
    model_trainer = ModelTrainer(data_processor)

    # Step 1: Load metadata
    print("Step 1: Loading metadata...")
    metadata_path = "dataset\\metadata.csv"
    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata file not found at {metadata_path}")
        # Try alternative paths
        metadata_path = "metadata.csv"
        if not os.path.exists(metadata_path):
            print("Error: Could not find metadata.csv file")
            return

    # Check if metadata is already loaded
    if data_processor.metadata is None:
        data_processor.load_metadata(metadata_path)
        print("Metadata loaded")
    else:
        print("Metadata already loaded")

    # Step 2: Create YOLO dataset structure with staining-based annotations
    print("\nStep 2: Creating YOLO dataset structure...")
    yolo_dir = os.path.join(data_processor.output_dir, "yolo_staining_dataset")
    completion_file = os.path.join(yolo_dir, "dataset_complete.txt")

    if os.path.exists(completion_file):
        print("Dataset already exists and is complete.")
        print(f"  Location: {yolo_dir}")

        _show_dataset_stats(yolo_dir)
    else:
        print("Creating new dataset...")
        yolo_dir = data_processor.prepare_staining_dataset()
        print(f"YOLO dataset created at: {yolo_dir}")


    data_processor.preview_bboxes()
    data_processor.validate_dataset_quality()

    # DO NOT ENABLE UNLESS YOU WANT TO SEE BOUNDING BOXES WHICH WILL PAUSE EXECUTION
    #data_processor.debug_data_pipeline()

    # Step 3: Train the model
    print("\nStep 3: Training YOLO model...")

    # Adjust training parameters based on dataset size and GPU capability
    training_params = {
        'epochs': 50,
        'imgsz': 640,
        #'batch': 16 if torch.cuda.is_available() else 4,
        'batch': 4,
        'model_type': 'yolov8n.pt'  # Can change to yolov8s.pt, yolov8m.pt for better performance
    }

    trained_model = model_trainer.train_yolo_model(**training_params)

    # Step 4: Verify model is saved and evaluate
    if trained_model is not None:
        # Check for automatically saved best model
        best_model_path = model_trainer.get_best_model_path()
        if os.path.exists(best_model_path):
            print(f"Best model automatically saved at: {best_model_path}")
        else:
            print("Best model not found at expected location")

        # Check for explicitly saved model
        explicit_model_path = model_trainer.get_explicit_model_path()
        if os.path.exists(explicit_model_path):
            print(f"Model explicitly saved at: {explicit_model_path}")

        # Test loading the model
        try:
            from ultralytics import YOLO
            loaded_model = YOLO(best_model_path if os.path.exists(best_model_path) else explicit_model_path)
            print("Model can be successfully loaded for inference")

            # Evaluate the trained model
            print("\nStep 4: Evaluating model performance...")
            metrics = model_trainer.evaluate_model()
            if metrics:
                print("Model evaluation completed successfully")

        except Exception as e:
            print(f"Error loading model: {e}")

        # Step 5: Test prediction on a sample image
        print("\nStep 5: Testing prediction on sample images...")
        test_samples = _find_test_samples(yolo_dir)
        for sample_path in test_samples[:2]:  # Test on first 2 samples
            if os.path.exists(sample_path):
                count, frame, detections = model_trainer.predict_cell_count(sample_path)
                print(f"Sample {os.path.basename(sample_path)}: {count} cells detected")
            else:
                print(f"Sample {sample_path} not found")

    else:
        print("Model training failed")

    print("\nFinished training pipeline.")
    evaluate_trained_model(data_processor)

def _show_dataset_stats(yolo_dir):
    """Show statistics about the existing dataset"""
    try:
        for split in ['train', 'val', 'test']:
            images_dir = os.path.join(yolo_dir, "images", split)
            labels_dir = os.path.join(yolo_dir, "labels", split)

            if os.path.exists(images_dir):
                image_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                label_count = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')]) if os.path.exists(labels_dir) else 0
                print(f"  {split}: {image_count} images, {label_count} labels")
    except Exception as e:
        print(f"  Could not read dataset stats: {e}")

def _find_test_samples(yolo_dir):
    """Find sample images for testing"""
    test_dir = os.path.join(yolo_dir, "images", "test")
    sample_images = []

    if os.path.exists(test_dir):
        for file in os.listdir(test_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                sample_images.append(os.path.join(test_dir, file))

    # If no test images, look in train or val directories
    if not sample_images:
        for split in ['train', 'val']:
            split_dir = os.path.join(yolo_dir, "images", split)
            if os.path.exists(split_dir):
                for file in os.listdir(split_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        sample_images.append(os.path.join(split_dir, file))
                        if len(sample_images) >= 2:  # Get 2 samples
                            break
                if sample_images:
                    break

    return sample_images

def evaluate_trained_model(data_processor):
    """Evaluate the trained model using the evaluator"""
    print("Running in TEST mode...")

    # Load metadata first
    metadata_path = "dataset\\metadata.csv"
    if not os.path.exists(metadata_path):
        metadata_path = "metadata.csv"
    data_processor.load_metadata(metadata_path)

    # Prepare dataset if not already done
    yolo_dir = data_processor.prepare_staining_dataset()

    # Initialize the evaluator
    evaluator = ModelEvaluator(data_processor)

    # Find model path directly
    model_path = evaluator._find_best_model_path()
    if not model_path:
        print("Error: No trained model found. Please train a model first.")
        return

    # Evaluate the model using the new evaluator interface
    print("\nStep 4: Evaluating model performance...")
    metrics = evaluator.evaluate_model(
        model_path=model_path,
        conf_threshold=0.3,
        iou_threshold=0.6,
        dataset_type='staining'
    )

    # Additional: Evaluate by staining type
    print("\nStep 5: Evaluating by staining type...")
    staining_results = evaluator.evaluate_by_staining_type(
        model_path=model_path,
        conf_threshold=0.3
    )

    print("\nStep 6: Analyzing failure cases...")
    failure_cases = evaluator.analyze_failure_cases(
        model_path=model_path,
        conf_threshold=0.3,
        num_cases=10
    )

    # Generate comprehensive report
    print("\nStep 7: Generating evaluation report...")
    if metrics:
        report_path = evaluator.generate_evaluation_report(metrics)

    print("Finished testing.")
    return

# Alternative main for quick testing
def quick_test():
    """Quick test function for development"""
    print("Running quick test...")

    # Initialize cell counting
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    data_processor = DataProcessor(data_root="dataset\\IDCIA", stainings=['AF488', 'DAPI', 'Cy3'])
    model_trainer = ModelTrainer(data_processor)

    # Load metadata
    data_processor.load_metadata("dataset\\metadata.csv")

    # Create dataset
    yolo_dir = data_processor.prepare_staining_dataset()

    data_processor.preview_bboxes()
    data_processor.validate_dataset_quality()

    # DO NOT ENABLE UNLESS YOU WANT TO SEE BOUNDING BOXES WHICH WILL PAUSE EXECUTION
    #data_processor.debug_data_pipeline()

    # Quick training with fewer epochs
    model = model_trainer.train_yolo_model(epochs=5, batch=4)

    if model:
        print("Quick test completed successfully!")
    else:
        print("Quick test failed!")
        return

    evaluate_trained_model(data_processor)


if __name__ == "__main__":
    # NOTE: CHANGE THIS FLAG TO SWITCH BETWEEN FULL PIPELINE AND QUICK TESTING
    # You can choose to run the full pipeline or quick test
    run_quick_test = True  # Set to True for quick testing, 5 epochs

    if run_quick_test:
        quick_test()
    else:
        main()