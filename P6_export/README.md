# P6 Export – Facial Expression Dataset

This project exports and uses the FER2013 facial expression dataset (happy, neutral, surprise) from Kaggle.

## Python version

**Use Python 3.10, 3.11, or 3.12.** TensorFlow does not support Python 3.14. On Apple Silicon, TensorFlow 2.12 is not available; the requirements use TensorFlow 2.13 instead (fully compatible).

On macOS with Homebrew:

```bash
brew install python@3.11
```

## Setup

1. **Create a virtual environment** with a supported Python (e.g. 3.11):

   ```bash
   cd P6_export
   python3.11 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r src/requirements.txt
   ```

3. **Download the Kaggle dataset** (choose one):

   **Option A – Using kagglehub (recommended)**  
   From `P6_export/src` with venv activated:

   ```bash
   cd src
   python download_dataset.py
   ```

   This uses [kagglehub](https://github.com/Kaggle/kagglehub) to download FER2013 and populates a `kaggle/` folder for `export_dataset.py`. You need Kaggle credentials (e.g. API key in `~/.kaggle/kaggle.json` or environment variables). Create an account at [Kaggle](https://www.kaggle.com) and get an API key from Account → Create New Token if needed.

   **Option B – Manual download**  
   - Download from: https://www.kaggle.com/datasets/msambare/fer2013  
   - Extract so that **inside** `src/` you have:

   ```
   P6_export/src/
   ├── kaggle/
   │   ├── train/
   │   │   ├── happy/
   │   │   ├── neutral/
   │   │   └── surprise/
   │   └── test/
   │       ├── happy/
   │       ├── neutral/
   │       └── surprise/
   ```

## Run

All commands are run from **`P6_export/src`** with the venv activated.

1. **Download the dataset** (if using kagglehub):  
   `python download_dataset.py`

2. **Export the first 5000 training images** (and all test images) for the target categories:

   ```bash
   cd src
   python export_dataset.py
   ```

   This creates/updates `train/` and `test/` under `src/` with the selected images.

3. **Inspect examples** for each category:

   ```bash
   python show_examples.py
   ```

   (The script is named `show_examples.py`, not `show_example.py`.) This opens a matplotlib window with sample images; close the window to exit.

## Summary

- Use Python 3.10–3.12 and a venv.
- Get FER2013 into `src/kaggle/`: run `download_dataset.py` (kagglehub) or place manual download there.
- Run `export_dataset.py` then `show_examples.py` from `src/`.
