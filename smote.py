import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

class CellCountSMOTE:
    """
    SMOTE-inspired augmentation for regression tasks (cell counting).
    Creates synthetic images for underrepresented cell count bins.
    """
    
    def __init__(self, bin_ranges=None, target_samples_per_bin=None, k_neighbors=5):
        """
        Initialize SMOTE augmentation.
        
        Args:
            bin_ranges: List of tuples defining count ranges, e.g., [(0,50), (50,100), (100,150)]
            target_samples_per_bin: Target number of samples per bin (uses max if None)
            k_neighbors: Number of neighbors for interpolation
        """
        self.bin_ranges = bin_ranges or [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 500)]
        self.target_samples_per_bin = target_samples_per_bin
        self.k_neighbors = k_neighbors
        
    def analyze_distribution(self, data_dir):
        """Analyze the distribution of cell counts across the dataset."""
        print("\n" + "="*80)
        print("ANALYZING CELL COUNT DISTRIBUTION")
        print("="*80)
        
        gt_dir = os.path.join(data_dir, "ground_truth")
        if not os.path.exists(gt_dir):
            raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
        
        # Count cells per image
        counts = []
        filenames = []
        for f in os.listdir(gt_dir):
            if f.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(gt_dir, f))
                    counts.append(len(df))
                    filenames.append(f.replace(".csv", ""))
                except pd.errors.EmptyDataError:
                    # Handle empty CSV files (0 cells)
                    counts.append(0)
                    filenames.append(f.replace(".csv", ""))
        
        data = pd.DataFrame({"filename": filenames, "cell_count": counts})
        
        # Bin the counts
        data['bin'] = pd.cut(data['cell_count'], 
                            bins=[b[0] for b in self.bin_ranges] + [self.bin_ranges[-1][1]],
                            labels=[f"{b[0]}-{b[1]}" for b in self.bin_ranges],
                            include_lowest=True)
        
        # Distribution summary
        print("\nCell Count Distribution by Bin:")
        bin_counts = data['bin'].value_counts().sort_index()
        for bin_label, count in bin_counts.items():
            print(f"  {bin_label:15s}: {count:4d} images")
        
        print(f"\nTotal images: {len(data)}")
        print(f"Mean cell count: {data['cell_count'].mean():.1f}")
        print(f"Median cell count: {data['cell_count'].median():.1f}")
        print(f"Std dev: {data['cell_count'].std():.1f}")
        
        # Plot distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(data['cell_count'], bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Cell Count')
        plt.ylabel('Frequency')
        plt.title('Distribution of Cell Counts')
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(1, 2, 2)
        bin_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.xlabel('Cell Count Bin')
        plt.ylabel('Number of Images')
        plt.title('Samples per Bin')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "count_distribution.png"))
        plt.close()
        
        return data, bin_counts
    
    def create_synthetic_image(self, img1, img2, alpha=0.5):
        """
        Create synthetic image by blending two images.
        
        Args:
            img1: First image (PIL Image or numpy array)
            img2: Second image (PIL Image or numpy array)
            alpha: Blending factor (0-1)
        
        Returns:
            Synthetic blended image
        """
        # Convert to numpy if needed
        if isinstance(img1, Image.Image):
            img1 = np.array(img1)
        if isinstance(img2, Image.Image):
            img2 = np.array(img2)
        
        # Ensure same dtype
        if img1.dtype != img2.dtype:
            img2 = img2.astype(img1.dtype)
        
        # Ensure same number of channels
        if len(img1.shape) != len(img2.shape):
            if len(img1.shape) == 3 and len(img2.shape) == 2:
                img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
            elif len(img1.shape) == 2 and len(img2.shape) == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        elif len(img1.shape) == 3 and img1.shape[2] != img2.shape[2]:
            # Different number of channels
            if img1.shape[2] == 3 and img2.shape[2] == 4:
                img2 = img2[:, :, :3]
            elif img1.shape[2] == 4 and img2.shape[2] == 3:
                img1 = img1[:, :, :3]
        
        # Ensure same size
        if img1.shape[:2] != img2.shape[:2]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Blend images using numpy (more robust than cv2.addWeighted)
        synthetic = (img1 * alpha + img2 * (1 - alpha)).astype(img1.dtype)
        
        return synthetic
    
    def interpolate_ground_truth(self, gt1, gt2, alpha=0.5):
        """
        Interpolate cell positions between two ground truth files.
        
        Args:
            gt1: First ground truth DataFrame
            gt2: Second ground truth DataFrame  
            alpha: Interpolation factor
        
        Returns:
            Interpolated ground truth DataFrame
        """
        # Handle empty ground truths
        if len(gt1) == 0 and len(gt2) == 0:
            # Both empty - return empty with proper columns
            return pd.DataFrame(columns=['X', 'Y'] if 'X' in gt1.columns or 'Y' in gt1.columns else gt1.columns)
        
        if len(gt1) == 0:
            return gt2.sample(n=len(gt2), replace=True)
        
        if len(gt2) == 0:
            return gt1.sample(n=len(gt1), replace=True)
        
        # Simple approach: blend coordinates
        n_cells = int(len(gt1) * alpha + len(gt2) * (1 - alpha))
        
        if n_cells == 0:
            # Return empty with proper columns
            return pd.DataFrame(columns=gt1.columns)
        
        # Sample cells from both ground truths
        n_from_1 = int(n_cells * alpha)
        n_from_2 = n_cells - n_from_1
        
        cells_from_1 = gt1.sample(n=min(n_from_1, len(gt1)), replace=True) if n_from_1 > 0 else pd.DataFrame(columns=gt1.columns)
        cells_from_2 = gt2.sample(n=min(n_from_2, len(gt2)), replace=True) if n_from_2 > 0 else pd.DataFrame(columns=gt2.columns)
        
        synthetic_gt = pd.concat([cells_from_1, cells_from_2], ignore_index=True)
        
        return synthetic_gt
    
    def augment_dataset(self, data_dir, output_dir, stain_type="train"):
        """
        Apply SMOTE-like augmentation to balance the dataset.
        
        Args:
            data_dir: Path to dataset directory (e.g., IDCIA/DAPI_train)
            output_dir: Path to save augmented dataset
            stain_type: Dataset split name for naming
        """
        print("\n" + "="*80)
        print(f"APPLYING SMOTE AUGMENTATION TO {data_dir}")
        print("="*80)
        
        # Analyze distribution
        data, bin_counts = self.analyze_distribution(data_dir)
        
        # Determine target samples per bin
        if self.target_samples_per_bin is None:
            self.target_samples_per_bin = int(bin_counts.max() * 1.2)  # 120% of max bin
        
        print(f"\nTarget samples per bin: {self.target_samples_per_bin}")
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "ground_truth"), exist_ok=True)
        
        # Copy original data
        print("\nCopying original data...")
        img_dir = os.path.join(data_dir, "images")
        gt_dir = os.path.join(data_dir, "ground_truth")
        
        for filename in tqdm(data['filename'], desc="Copying originals"):
            shutil.copy(
                os.path.join(img_dir, f"{filename}.tiff"),
                os.path.join(output_dir, "images", f"{filename}.tiff")
            )
            shutil.copy(
                os.path.join(gt_dir, f"{filename}.csv"),
                os.path.join(output_dir, "ground_truth", f"{filename}.csv")
            )
        
        # Generate synthetic samples for underrepresented bins
        print("\nGenerating synthetic samples...")
        synthetic_count = 0
        
        for bin_label in bin_counts.index:
            current_count = bin_counts[bin_label]
            needed = self.target_samples_per_bin - current_count
            
            if needed <= 0:
                continue
            
            print(f"\n  Bin {bin_label}: {current_count} samples, generating {needed} synthetic")
            
            # Get samples from this bin
            bin_samples = data[data['bin'] == bin_label]
            
            if len(bin_samples) < 2:
                print(f"    Skipping (need at least 2 samples for interpolation)")
                continue
            
            # Find nearest neighbors for each sample
            feature_matrix = bin_samples[['cell_count']].values
            nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(bin_samples))).fit(feature_matrix)
            
            for _ in tqdm(range(needed), desc=f"  Generating for {bin_label}"):
                # Random sample from bin
                idx = np.random.randint(len(bin_samples))
                sample = bin_samples.iloc[idx]
                
                # Find neighbors
                distances, indices = nbrs.kneighbors([feature_matrix[idx]])
                neighbor_idx = np.random.choice(indices[0][1:])  # Exclude self
                neighbor = bin_samples.iloc[neighbor_idx]
                
                # Random interpolation factor
                alpha = np.random.uniform(0.3, 0.7)
                
                # Load images
                img1 = Image.open(os.path.join(img_dir, f"{sample['filename']}.tiff"))
                img2 = Image.open(os.path.join(img_dir, f"{neighbor['filename']}.tiff"))
                
                # Load ground truths
                gt1 = pd.read_csv(os.path.join(gt_dir, f"{sample['filename']}.csv"))
                gt2 = pd.read_csv(os.path.join(gt_dir, f"{neighbor['filename']}.csv"))
                
                # Create synthetic image and ground truth
                synthetic_img = self.create_synthetic_image(img1, img2, alpha)
                synthetic_gt = self.interpolate_ground_truth(gt1, gt2, alpha)
                
                # Save synthetic sample
                synthetic_filename = f"synthetic_{stain_type}_{synthetic_count}"
                
                # Handle different image types (uint8, uint16, etc.)
                if synthetic_img.dtype == np.uint16:
                    # For 16-bit images, use tifffile or PIL with mode='I;16'
                    if len(synthetic_img.shape) == 3:
                        # Multi-channel 16-bit - save each channel or convert
                        # Convert to 8-bit for compatibility
                        synthetic_img_8bit = (synthetic_img / 256).astype(np.uint8)
                        Image.fromarray(synthetic_img_8bit).save(
                            os.path.join(output_dir, "images", f"{synthetic_filename}.tiff")
                        )
                    else:
                        # Single channel 16-bit
                        Image.fromarray(synthetic_img, mode='I;16').save(
                            os.path.join(output_dir, "images", f"{synthetic_filename}.tiff")
                        )
                else:
                    # Standard 8-bit images
                    Image.fromarray(synthetic_img).save(
                        os.path.join(output_dir, "images", f"{synthetic_filename}.tiff")
                    )
                
                synthetic_gt.to_csv(
                    os.path.join(output_dir, "ground_truth", f"{synthetic_filename}.csv"),
                    index=False
                )
                
                synthetic_count += 1
        
        print(f"\n✓ Augmentation complete!")
        print(f"  Original samples: {len(data)}")
        print(f"  Synthetic samples: {synthetic_count}")
        print(f"  Total samples: {len(data) + synthetic_count}")
        print(f"  Output directory: {output_dir}")
        
        # Analyze augmented distribution
        print("\n" + "="*80)
        print("AUGMENTED DATASET DISTRIBUTION")
        print("="*80)
        self.analyze_distribution(output_dir)


def main():
    """
    Main function to run SMOTE augmentation on IDCIA dataset.
    """
    
    # Configuration
    BASE_DIR = "IDCIA"
    OUTPUT_BASE = "IDCIA_augmented"
    
    # Define cell count bins (adjust based on your data)
    BIN_RANGES = [
        (0, 50),
        (50, 100),
        (100, 150),
        (150, 200),
        (200, 300),
        (300, 500)
    ]
    
    # Initialize SMOTE
    smote = CellCountSMOTE(
        bin_ranges=BIN_RANGES,
        target_samples_per_bin=None,  # Will use 120% of max bin
        k_neighbors=5
    )
    
    # Find all staining types
    stain_dirs = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    stain_types = sorted(set('_'.join(d.split('_')[:-1]) for d in stain_dirs if '_' in d))
    
    print(f"Found staining types: {', '.join(stain_types)}")
    
    # Apply SMOTE to each stain type and split
    for stain in stain_types:
        for split in ['train', 'val', 'test']:
            data_dir = os.path.join(BASE_DIR, f"{stain}_{split}")
            
            if not os.path.exists(data_dir):
                print(f"\nSkipping {stain}_{split} (directory not found)")
                continue
            
            output_dir = os.path.join(OUTPUT_BASE, f"{stain}_{split}")
            
            print(f"\n{'='*80}")
            print(f"Processing: {stain}_{split}")
            print(f"{'='*80}")
            
            smote.augment_dataset(data_dir, output_dir, stain_type=f"{stain}_{split}")
    

    print(f"Augmented datasets saved in: {OUTPUT_BASE}/")


if __name__ == "__main__":
    main()