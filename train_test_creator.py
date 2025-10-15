import os
import shutil
from sklearn.model_selection import train_test_split

def organize_idcia_dataset(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Organizes images (.tiff) and ground truths (.csv) into train/val/test structure.
    Assumes this script is run from within COMS5710/, and that:
        ./img/
        ./ground_truth/
    already exist.
    """

    base_dir = os.getcwd()
    img_dir = os.path.join(base_dir, "img")
    gt_dir = os.path.join(base_dir, "ground_truth")

    # Check that both source folders exist
    if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
        raise FileNotFoundError(f"Could not find 'img' or 'ground_truth' in {base_dir}")

    # Destination directories
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(base_dir, "IDCIA", split, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "IDCIA", split, "ground_truth"), exist_ok=True)

    # Collect all valid image-ground truth pairs
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".tiff")])
    csv_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".csv")])

    ids = [os.path.splitext(f)[0] for f in img_files if f.replace(".tiff", ".csv") in csv_files]
    print(f"Found {len(ids)} valid image/ground_truth pairs.")

    # Split into train (80%) and temp (20%)
    train_ids, temp_ids = train_test_split(ids, test_size=(1 - train_ratio), random_state=random_state)

    # Split temp into val (50%) and test (50%) of temp (10% each)
    val_ids, test_ids = train_test_split(temp_ids, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=random_state)

    def copy_files(id_list, split):
        img_dest = os.path.join(base_dir, "IDCIA", split, "images")
        gt_dest = os.path.join(base_dir, "IDCIA", split, "ground_truth")
        for id_ in id_list:
            shutil.copy(os.path.join(img_dir, f"{id_}.tiff"), os.path.join(img_dest, f"{id_}.tiff"))
            shutil.copy(os.path.join(gt_dir, f"{id_}.csv"), os.path.join(gt_dest, f"{id_}.csv"))

    copy_files(train_ids, "train")
    copy_files(val_ids, "val")
    copy_files(test_ids, "test")

    print("\n✅ Dataset organized successfully inside ./IDCIA")
    print(f"Train: {len(train_ids)} images")
    print(f"Val:   {len(val_ids)} images")
    print(f"Test:  {len(test_ids)} images")

if __name__ == "__main__":
    organize_idcia_dataset()
