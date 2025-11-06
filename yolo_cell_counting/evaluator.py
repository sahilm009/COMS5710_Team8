import os
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import pandas as pd
from collections import defaultdict
import yaml

class ModelEvaluator:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.output_dir = data_processor.output_dir

    def evaluate_model(self, model_path=None, conf_threshold=0.3, iou_threshold=0.6, dataset_type='staining'):
        """
        Evaluate YOLO model using proper object detection metrics

        Args:
            model_path (str): Path to trained model weights
            conf_threshold (float): Confidence threshold for detection
            iou_threshold (float): IoU threshold for mAP calculation
            dataset_type (str): 'staining' or 'celltype' dataset to evaluate

        Returns:
            dict: Comprehensive evaluation results
        """
        print("Evaluating model with proper object detection metrics...")

        # Find model if not specified
        if model_path is None:
            model_path = self._find_best_model_path()

        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return None

        # Load model
        model = YOLO(model_path)
        print(f"Loaded model: {os.path.basename(model_path)}")

        # Get dataset path
        data_yaml = self._get_data_yaml_path(dataset_type)
        if not os.path.exists(data_yaml):
            print(f"Error: Data YAML not found: {data_yaml}")
            return None

        # Run YOLO validation for proper metrics
        print("Running YOLO validation...")
        metrics = model.val(
            data=data_yaml,
            split='test',
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=True
        )

        # Extract and analyze metrics
        results = self._extract_comprehensive_metrics(metrics, model_path, dataset_type)

        # Additional detailed analysis
        self._perform_detailed_analysis(results, dataset_type)

        return results

    def _get_data_yaml_path(self, dataset_type):
        """Get the correct data.yaml path for the dataset type"""
        if dataset_type == 'staining':
            return os.path.join(self.output_dir, "yolo_staining_dataset", "data.yaml")
        elif dataset_type == 'celltype':
            return os.path.join(self.output_dir, "yolo_celltype_dataset", "data.yaml")
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def _extract_comprehensive_metrics(self, metrics, model_path, dataset_type):
        """Extract comprehensive metrics from YOLO validation results"""

        # Get class names
        if dataset_type == 'staining':
            class_names = getattr(self.data_processor, 'stainings', ['AF488', 'DAPI', 'Cy3'])
        else:
            # Load from data.yaml for cell type
            data_yaml_path = self._get_data_yaml_path(dataset_type)
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            class_names = data_config.get('names', [])

        results = {
            'model_path': model_path,
            'dataset_type': dataset_type,
            'class_names': class_names,
            'overall_metrics': {},
            'per_class_metrics': {},
            'training_analysis': {},
            'recommendations': []
        }

        # Extract overall metrics
        if hasattr(metrics, 'box'):
            results['overall_metrics'] = {
                'mAP50': getattr(metrics.box, 'map50', 0),
                'mAP50_95': getattr(metrics.box, 'map', 0),
                'precision': getattr(metrics.box, 'mp', 0),
                'recall': getattr(metrics.box, 'mr', 0),
                'f1_score': self._calculate_f1_score(
                    getattr(metrics.box, 'mp', 0),
                    getattr(metrics.box, 'mr', 0)
                )
            }

        # Extract per-class metrics
        if hasattr(metrics, 'box') and hasattr(metrics.box, 'maps'):
            for i, class_name in enumerate(class_names):
                if i < len(metrics.box.maps):
                    results['per_class_metrics'][class_name] = {
                        'mAP50_95': metrics.box.maps[i],
                        'precision': getattr(metrics.box, 'p', [0] * len(class_names))[i] if hasattr(metrics.box, 'p') else 0,
                        'recall': getattr(metrics.box, 'r', [0] * len(class_names))[i] if hasattr(metrics.box, 'r') else 0,
                        'images': getattr(metrics, 'nt', [0] * len(class_names))[i] if hasattr(metrics, 'nt') else 0,
                        'instances': getattr(metrics, 'ni', [0] * len(class_names))[i] if hasattr(metrics, 'ni') else 0
                    }

        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)

        return results

    def _calculate_f1_score(self, precision, recall):
        """Calculate F1 score from precision and recall"""
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def _generate_recommendations(self, results):
        """Generate specific recommendations based on performance"""
        recommendations = []
        overall = results['overall_metrics']
        per_class = results['per_class_metrics']

        # Overall performance recommendations
        if overall['mAP50_95'] < 0.1:
            recommendations.append("CRITICAL: Overall mAP very low - check data quality and training process")
        elif overall['mAP50_95'] < 0.3:
            recommendations.append("MODERATE: Model needs improvement - consider more training data or augmentation")
        elif overall['mAP50_95'] < 0.5:
            recommendations.append("GOOD: Decent performance - fine-tuning could improve further")
        else:
            recommendations.append("EXCELLENT: Great performance!")

        # Per-class recommendations
        for class_name, metrics in per_class.items():
            if metrics['mAP50_95'] < 0.05:
                recommendations.append(f"{class_name}: Very poor performance (mAP: {metrics['mAP50_95']:.3f}) - check class imbalance and data quality")
            elif metrics['mAP50_95'] < 0.2:
                recommendations.append(f"{class_name}: Poor performance (mAP: {metrics['mAP50_95']:.3f}) - needs more samples or augmentation")
            elif metrics['recall'] < 0.1:
                recommendations.append(f"{class_name}: Very low recall ({metrics['recall']:.3f}) - model misses most cells")
            elif metrics['precision'] < 0.3:
                recommendations.append(f"{class_name}: Low precision ({metrics['precision']:.3f}) - many false positives")

        return recommendations

    def _perform_detailed_analysis(self, results, dataset_type):
        """Perform and print detailed analysis of results"""
        print("\n" + "="*80)
        print("DETAILED PERFORMANCE ANALYSIS")
        print("="*80)

        # Overall metrics
        overall = results['overall_metrics']
        print(f"\nOVERALL METRICS:")
        print(f"   mAP@50:       {overall.get('mAP50', 0):.4f}")
        print(f"   mAP@50-95:    {overall.get('mAP50_95', 0):.4f}")
        print(f"   Precision:    {overall.get('precision', 0):.4f}")
        print(f"   Recall:       {overall.get('recall', 0):.4f}")
        print(f"   F1 Score:     {overall.get('f1_score', 0):.4f}")

        # Performance interpretation
        mAP = overall.get('mAP50_95', 0)
        if mAP >= 0.5:
            performance_level = "EXCELLENT"
        elif mAP >= 0.3:
            performance_level = "GOOD"
        elif mAP >= 0.15:
            performance_level = "MODERATE"
        elif mAP >= 0.05:
            performance_level = "POOR"
        else:
            performance_level = "VERY POOR"

        print(f"   Performance:  {performance_level}")

        # Per-class metrics
        print(f"\nPER-CLASS PERFORMANCE:")
        print("-" * 70)
        print(f"{'Class':<12} {'Images':<8} {'Instances':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10} {'Status':<12}")
        print("-" * 70)

        for class_name, metrics in results['per_class_metrics'].items():
            # Determine status
            if metrics['mAP50_95'] >= 0.3:
                status = "GOOD"
            elif metrics['mAP50_95'] >= 0.15:
                status = "MODERATE"
            elif metrics['mAP50_95'] >= 0.05:
                status = "NEEDS WORK"
            else:
                status = "POOR"

            print(f"{class_name:<12} {metrics['images']:<8} {metrics['instances']:<10} "
                  f"{metrics['mAP50_95']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {status:<12}")

        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        print("-" * 70)
        for i, recommendation in enumerate(results['recommendations'], 1):
            print(f"{i:2d}. {recommendation}")

    def evaluate_by_staining_type(self, model_path=None, conf_threshold=0.3):
        """Evaluate performance separately for each staining type"""
        print("\n" + "="*80)
        print("STAINING-SPECIFIC EVALUATION")
        print("="*80)

        return self.evaluate_model(model_path, conf_threshold, 0.6, 'staining')

    def evaluate_by_cell_type(self, model_path=None, conf_threshold=0.3):
        """Evaluate performance for cell type classification"""
        print("\n" + "="*80)
        print("CELL TYPE EVALUATION")
        print("="*80)

        return self.evaluate_model(model_path, conf_threshold, 0.6, 'celltype')

    def compare_multiple_models(self, model_paths, conf_threshold=0.3):
        """Compare multiple models and rank their performance"""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)

        comparison_results = {}

        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"\nEvaluating: {os.path.basename(model_path)}")
                results = self.evaluate_model(model_path, conf_threshold)
                if results:
                    comparison_results[model_path] = results

        # Rank models by mAP@50-95
        ranked_models = sorted(
            comparison_results.items(),
            key=lambda x: x[1]['overall_metrics'].get('mAP50_95', 0),
            reverse=True
        )

        print(f"\nMODEL RANKING (by mAP@50-95):")
        print("-" * 70)
        print(f"{'Rank':<6} {'Model':<30} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}")
        print("-" * 70)

        for rank, (model_path, results) in enumerate(ranked_models, 1):
            overall = results['overall_metrics']
            model_name = os.path.basename(model_path)
            print(f"{rank:<6} {model_name:<30} {overall.get('mAP50_95', 0):<10.4f} "
                  f"{overall.get('precision', 0):<10.4f} {overall.get('recall', 0):<10.4f}")

        return comparison_results

    def analyze_failure_cases(self, model_path=None, conf_threshold=0.3, num_cases=10):
        """Analyze specific failure cases with detailed information"""
        print("\n" + "="*80)
        print("FAILURE CASE ANALYSIS")
        print("="*80)

        if model_path is None:
            model_path = self._find_best_model_path()

        model = YOLO(model_path)
        test_images_dir = os.path.join(self.output_dir, "yolo_staining_dataset", "images", "test")
        test_labels_dir = os.path.join(self.output_dir, "yolo_staining_dataset", "labels", "test")

        if not os.path.exists(test_images_dir):
            print(f"Error: Test directory not found: {test_images_dir}")
            return None

        test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        errors = []

        print(f"Analyzing {min(num_cases, len(test_images))} test cases...")

        for img_file in tqdm(test_images[:num_cases], desc="Analyzing failures"):
            img_path = os.path.join(test_images_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            label_path = os.path.join(test_labels_dir, base_name + ".txt")

            # Get ground truth
            true_annotations = self._parse_yolo_labels(label_path)
            true_count = len(true_annotations)

            # Get predictions
            results = model.predict(source=img_path, conf=conf_threshold, verbose=False)
            result = results[0]

            pred_count = len(result.boxes) if result.boxes else 0
            error = abs(pred_count - true_count)

            if true_count > 0:
                error_ratio = error / true_count
            else:
                error_ratio = pred_count  # All predictions are false positives if no ground truth

            errors.append({
                'image': img_file,
                'true_count': true_count,
                'pred_count': pred_count,
                'error': error,
                'error_ratio': error_ratio,
                'staining_type': self._extract_staining_type(img_file)
            })

        # Sort by error ratio (worst first)
        errors.sort(key=lambda x: x['error_ratio'], reverse=True)

        print(f"\nTOP {min(5, len(errors))} FAILURE CASES (Highest Error Ratio):")
        print("-" * 80)
        print(f"{'Image':<20} {'Staining':<10} {'True':<6} {'Pred':<6} {'Error':<8} {'Error Ratio':<12}")
        print("-" * 80)

        for error in errors[:5]:
            print(f"{error['image'][:18]:<20} {error['staining_type']:<10} "
                  f"{error['true_count']:<6} {error['pred_count']:<6} "
                  f"{error['error']:<8} {error['error_ratio']:<12.2f}")

        return errors

    def _find_best_model_path(self):
        """Find the best model path from training outputs"""
        possible_locations = [
            os.path.join(self.output_dir, 'cell_staining_classification', 'weights', 'best.pt'),
            os.path.join(self.output_dir, 'train', 'weights', 'best.pt'),
            os.path.join(self.output_dir, 'weights', 'best.pt'),
            os.path.join(self.output_dir, 'best.pt'),
        ]

        for path in possible_locations:
            if os.path.exists(path):
                return path

        # Search recursively
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                if file == 'best.pt' and file.endswith('.pt'):
                    return os.path.join(root, file)

        print("Error: No trained model found. Please train a model first.")
        return None

    def _parse_yolo_labels(self, label_path):
        """Parse YOLO format label file"""
        annotations = []
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            class_id = int(parts[0])
                            annotations.append({'class_id': class_id})
            except Exception as e:
                print(f"Warning: Error parsing label file {label_path}: {e}")
        return annotations

    def _extract_staining_type(self, filename):
        """Extract staining type from filename"""
        stainings = getattr(self.data_processor, 'stainings', ['AF488', 'DAPI', 'Cy3'])
        for staining in stainings:
            if staining.lower() in filename.lower():
                return staining
        return 'unknown'

    def generate_evaluation_report(self, results, output_path=None):
        """Generate a comprehensive evaluation report"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'evaluation_report.txt')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("YOLO MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Overall metrics
            overall = results['overall_metrics']
            f.write("OVERALL PERFORMANCE:\n")
            f.write(f"mAP@50:       {overall.get('mAP50', 0):.4f}\n")
            f.write(f"mAP@50-95:    {overall.get('mAP50_95', 0):.4f}\n")
            f.write(f"Precision:    {overall.get('precision', 0):.4f}\n")
            f.write(f"Recall:       {overall.get('recall', 0):.4f}\n")
            f.write(f"F1 Score:     {overall.get('f1_score', 0):.4f}\n\n")

            # Per-class performance
            f.write("PER-CLASS PERFORMANCE:\n")
            for class_name, metrics in results['per_class_metrics'].items():
                f.write(f"{class_name}: mAP={metrics['mAP50_95']:.4f}, "
                       f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
                       f"Instances={metrics['instances']}\n")

            # Recommendations
            f.write("\nRECOMMENDATIONS:\n")
            for i, rec in enumerate(results['recommendations'], 1):
                f.write(f"{i}. {rec}\n")

        print(f"Evaluation report saved to: {output_path}")
        return output_path

    def generate_prediction_ground_truth_ratio_to_csv(self, model_path=None, conf_threshold=0.3):
        """Generate a CSV file comparing prediction counts to ground truth counts"""
        print("\n" + "="*80)
        print("PREDICTION vs GROUND TRUTH COUNT ANALYSIS")
        print("="*80)

        if model_path is None:
            model_path = self._find_best_model_path()

        model = YOLO(model_path)
        test_images_dir = os.path.join(self.output_dir, "yolo_staining_dataset", "images", "test")
        test_labels_dir = os.path.join(self.output_dir, "yolo_staining_dataset", "labels", "test")

        if not os.path.exists(test_images_dir):
            print(f"Error: Test directory not found: {test_images_dir}")
            return None

        test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        records = []

        print(f"Analyzing {len(test_images)} test images...")

        for img_file in tqdm(test_images, desc="Processing images"):
            img_path = os.path.join(test_images_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            label_path = os.path.join(test_labels_dir, base_name + ".txt")

            # Get ground truth
            true_annotations = self._parse_yolo_labels(label_path)
            true_count = len(true_annotations)

            # Get predictions
            results = model.predict(source=img_path, conf=conf_threshold, verbose=False)
            result = results[0]

            pred_count = len(result.boxes) if result.boxes else 0

            records.append({
                'pred_count': pred_count,
                'true_count': true_count,
                'staining_type': self._extract_staining_type(img_file),
                'img_file': img_file
            })

        # Save to CSV
        df = pd.DataFrame(records)
        output_csv = os.path.join(self.output_dir, 'prediction_vs_ground_truth_counts.csv')
        df.to_csv(output_csv, index=False)

        print(f"Prediction vs Ground Truth count analysis saved to: {output_csv}")
        return output_csv