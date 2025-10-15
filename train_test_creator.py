import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def organize_idcia(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Organizes images (.tiff) and ground truths (.csv) into train/val/test splits
    for each unique staining type found in metadata.csv.

    Directory structure created:
      IDCIA/
        ├── Cy3_train/
        ├── Cy3_val/
        ├── Cy3_test/
        ├── AF488_train/
        ├── AF488_val/
        └── AF488_test/
    """

    base_dir = os.getcwd()
    img_dir = os.path.join(base_dir, "img")
    gt_dir = os.path.join(base_dir, "ground_truth")
    metadata_path = os.path.join(base_dir, "metadata.csv")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Could not find metadata.csv in {base_dir}")
    if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
        raise FileNotFoundError(f"Could not find 'img' or 'ground_truth' in {base_dir}")

    # Load metadata
    meta = pd.read_csv(metadata_path)
    if "id" not in meta.columns or "staining" not in meta.columns:
        raise ValueError("metadata.csv must contain 'id' and 'staining' columns.")

    # Drop rows with missing IDs
    meta = meta.dropna(subset=["id", "staining"])
    meta["id"] = meta["id"].astype(str)

    stainings = meta["staining"].unique()
    print(f"Found {len(stainings)} unique staining types: {', '.join(stainings)}")

    for stain in stainings:
        subset = meta[meta["staining"] == stain]
        ids = subset["id"].tolist()

        # Skip if no matching files exist
        valid_ids = [i for i in ids if os.path.exists(os.path.join(img_dir, f"{i}.tiff"))
                     and os.path.exists(os.path.join(gt_dir, f"{i}.csv"))]
        if not valid_ids:
            print(f"Skipping {stain} (no valid image/CSV pairs found).")
            continue

        # Split into train, val, test
        train_ids, temp_ids = train_test_split(valid_ids, test_size=(1 - train_ratio), random_state=random_state)
        val_ids, test_ids = train_test_split(temp_ids, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=random_state)

        # Destination dirs per staining + split
        for split_name, id_list in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
            stain_split_dir = os.path.join(base_dir, "IDCIA", f"{stain}_{split_name}")
            os.makedirs(os.path.join(stain_split_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(stain_split_dir, "ground_truth"), exist_ok=True)

            for id_ in id_list:
                src_img = os.path.join(img_dir, f"{id_}.tiff")
                src_gt = os.path.join(gt_dir, f"{id_}.csv")
                dst_img = os.path.join(stain_split_dir, "images", f"{id_}.tiff")
                dst_gt = os.path.join(stain_split_dir, "ground_truth", f"{id_}.csv")

                if os.path.exists(src_img) and os.path.exists(src_gt):
                    shutil.copy(src_img, dst_img)
                    shutil.copy(src_gt, dst_gt)

            print(f"{stain} - {split_name}: {len(id_list)} images")

    print("\nDataset organized successfully inside ./IDCIA by staining type")

if __name__ == "__main__":
    organize_idcia()
