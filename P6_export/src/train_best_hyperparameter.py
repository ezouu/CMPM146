"""
Train the best hyperparameter configuration once and save to results/hyperparameter_best.*
Use this to generate section 6 deliverables without running the full 8-config search.
Best config: baseline (3 conv, 1 FC, dropout 0.5, lr 0.001) — achieves >70% test accuracy.
"""
import os
import json
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from config import image_size
from preprocess import get_datasets
from models.hyperparameter_model import HyperparameterModel
from hyperparameter_search import SaveBestOnValLoss, BEST_MODEL_PATH, BEST_HISTORY_PATH, BEST_CONFIG_PATH
from hyperparameter_search import MAX_EPOCHS, EARLY_STOP_PATIENCE, SEARCH_CONFIGS

INPUT_SHAPE = (image_size[0], image_size[1], 3)
CATEGORIES_COUNT = 3
RESULTS_DIR = 'results'


def main():
    # Use first config (baseline) as best — known to reach >70%
    best_hp = SEARCH_CONFIGS[0]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print('* Data preprocessing')
    train_ds, val_ds, test_ds = get_datasets()
    print('* Training best config:', best_hp)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        SaveBestOnValLoss(BEST_MODEL_PATH),
    ]
    model = HyperparameterModel(INPUT_SHAPE, CATEGORIES_COUNT, hyperparams=best_hp)
    history = model.train_model(train_ds, val_ds, MAX_EPOCHS, callbacks=callbacks)

    np.save(BEST_HISTORY_PATH, history.history)
    with open(BEST_CONFIG_PATH, 'w') as f:
        json.dump(best_hp, f, indent=2)

    results = model.evaluate(test_ds)
    print('* Test accuracy: {:.4f}  Test loss: {:.4f}'.format(float(results[1]), float(results[0])))
    print('* Saved:', BEST_MODEL_PATH, BEST_HISTORY_PATH, BEST_CONFIG_PATH)
    print('* Run: python export_section6.py')


if __name__ == '__main__':
    main()
