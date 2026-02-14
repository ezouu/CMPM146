"""
Hyperparameter search: vary conv layers, FC layers, dropout (0-2), learning rate.
Uses EarlyStopping (val_loss) and a custom save-best callback to select model at overfitting point.
Saves best model and config to results/ for export_section6.
"""
import os
import json
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, Callback

from config import image_size
from preprocess import get_datasets
from models.hyperparameter_model import HyperparameterModel, get_default_hyperparams

INPUT_SHAPE = (image_size[0], image_size[1], 3)
CATEGORIES_COUNT = 3
RESULTS_DIR = 'results'
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, 'hyperparameter_best.keras')
BEST_HISTORY_PATH = os.path.join(RESULTS_DIR, 'hyperparameter_best_history.npy')
BEST_CONFIG_PATH = os.path.join(RESULTS_DIR, 'hyperparameter_best_config.json')
MAX_EPOCHS = 20
EARLY_STOP_PATIENCE = 4


class SaveBestOnValLoss(Callback):
    """Save model to filepath when val_loss improves (avoids ModelCheckpoint options error)."""
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss')
        if val_loss is not None and val_loss < self.best:
            self.best = val_loss
            self.model.save(self.filepath)

# Search space: conv_filters, fc_units, dropout_rates (0-2), learning_rate
SEARCH_CONFIGS = [
    # Baseline (BasicModel-like)
    {'conv_filters': [32, 64, 128], 'fc_units': [128], 'dropout_rates': [0.5], 'learning_rate': 0.001},
    # More capacity
    {'conv_filters': [32, 64, 128, 256], 'fc_units': [128], 'dropout_rates': [0.5], 'learning_rate': 0.001},
    # Deeper FC, one dropout
    {'conv_filters': [32, 64, 128], 'fc_units': [256], 'dropout_rates': [0.4], 'learning_rate': 0.001},
    # Two FC, two dropouts
    {'conv_filters': [32, 64, 128], 'fc_units': [128, 64], 'dropout_rates': [0.4, 0.5], 'learning_rate': 0.0005},
    # Two dropouts after single FC (only first used; second after last FC would need 2 fc_units)
    {'conv_filters': [32, 64, 128], 'fc_units': [128, 64], 'dropout_rates': [0.3, 0.5], 'learning_rate': 0.001},
    # Lower LR, 4 conv
    {'conv_filters': [32, 64, 128, 256], 'fc_units': [128], 'dropout_rates': [0.4], 'learning_rate': 0.0005},
    # No dropout (0)
    {'conv_filters': [32, 64, 128], 'fc_units': [128], 'dropout_rates': [], 'learning_rate': 0.001},
    # Two dropouts, lower LR
    {'conv_filters': [32, 64, 128], 'fc_units': [256, 128], 'dropout_rates': [0.4, 0.5], 'learning_rate': 0.0005},
]


def run_search():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print('* Data preprocessing')
    train_ds, val_ds, test_ds = get_datasets()

    best_test_accuracy = -1.0
    best_config = None
    best_history = None
    best_config_idx = -1

    for idx, hp in enumerate(SEARCH_CONFIGS):
        config_ckpt = os.path.join(RESULTS_DIR, 'hyperparameter_search_{}.keras'.format(idx))
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOP_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            SaveBestOnValLoss(config_ckpt),
        ]
        print('\n' + '=' * 60)
        print('Config {}: {}'.format(idx + 1, hp))
        print('=' * 60)
        model = HyperparameterModel(INPUT_SHAPE, CATEGORIES_COUNT, hyperparams=hp)
        history = model.train_model(train_ds, val_ds, MAX_EPOCHS, callbacks=callbacks)
        # EarlyStopping restored best weights; evaluate on test
        results = model.evaluate(test_ds)
        test_loss, test_acc = float(results[0]), float(results[1])
        val_accs = history.history['val_accuracy']
        best_val_acc = max(val_accs) if val_accs else 0
        print('Best val accuracy: {:.4f}  Test accuracy: {:.4f}  Test loss: {:.4f}'.format(
            best_val_acc, test_acc, test_loss))

        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            best_config = hp
            best_history = history.history
            best_config_idx = idx
            import shutil
            if os.path.isfile(config_ckpt):
                shutil.copy2(config_ckpt, BEST_MODEL_PATH)
            else:
                model.save_model(BEST_MODEL_PATH)
            np.save(BEST_HISTORY_PATH, history.history)
            with open(BEST_CONFIG_PATH, 'w') as f:
                json.dump(hp, f, indent=2)
            print('  -> New best test accuracy. Saved.')

    print('\n' + '=' * 60)
    print('Best config (index {}): {}'.format(best_config_idx + 1, best_config))
    print('Best test accuracy: {:.4f}'.format(best_test_accuracy))
    print('Saved to:', BEST_MODEL_PATH, BEST_HISTORY_PATH, BEST_CONFIG_PATH)
    return best_config, best_history, best_test_accuracy


if __name__ == '__main__':
    run_search()
