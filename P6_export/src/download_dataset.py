"""
Download FER2013 dataset using kagglehub and prepare it for export_dataset.py.
Creates a 'kaggle' directory with train/ and test/ subdirs (happy, neutral, surprise).
"""
import kagglehub
import os
import shutil

KAGGLE_DIR = "kaggle"
DATASET = "msambare/fer2013"


def main():
    print(f"Downloading {DATASET} via kagglehub...")
    path = kagglehub.dataset_download(DATASET)
    print("Path to dataset files:", path)

    # Discover structure: path may contain train/ and test/ at top level or one level down
    entries = os.listdir(path)
    train_src = None
    test_src = None

    if "train" in entries and "test" in entries:
        train_src = os.path.join(path, "train")
        test_src = os.path.join(path, "test")
    else:
        # Some datasets have a single subdir (e.g. fer2013) containing train/test
        for name in entries:
            subpath = os.path.join(path, name)
            if not os.path.isdir(subpath):
                continue
            sub_entries = os.listdir(subpath)
            if "train" in sub_entries and "test" in sub_entries:
                train_src = os.path.join(subpath, "train")
                test_src = os.path.join(subpath, "test")
                break

    if train_src is None or test_src is None:
        raise RuntimeError(
            f"Could not find train/ and test/ under {path!r}. "
            f"Top-level entries: {entries!r}. "
            "Please place the dataset manually in a directory named 'kaggle' with train/ and test/."
        )

    os.makedirs(KAGGLE_DIR, exist_ok=True)
    train_dst = os.path.join(KAGGLE_DIR, "train")
    test_dst = os.path.join(KAGGLE_DIR, "test")

    if os.path.exists(train_dst):
        shutil.rmtree(train_dst)
    if os.path.exists(test_dst):
        shutil.rmtree(test_dst)

    shutil.copytree(train_src, train_dst)
    shutil.copytree(test_src, test_dst)
    print(f"Dataset copied into '{KAGGLE_DIR}/' (train and test).")
    print("You can now run: python export_dataset.py")


if __name__ == "__main__":
    main()
