"""
Configurable model for hyperparameter search.
Hyperparameters: conv_filters, fc_units, dropout_rates, learning_rate.
"""
from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.legacy import RMSprop as LegacyRMSprop


def get_default_hyperparams():
    """Default hyperparameter dict (matches BasicModel-style config)."""
    return {
        'conv_filters': [32, 64, 128],
        'fc_units': [128],
        'dropout_rates': [0.5],
        'learning_rate': 0.001,
    }


class HyperparameterModel(Model):
    """Model built from hyperparameter dict."""

    def __init__(self, input_shape, categories_count, hyperparams=None):
        self._hyperparams = hyperparams or get_default_hyperparams()
        super().__init__(input_shape, categories_count)

    def _define_model(self, input_shape, categories_count):
        hp = self._hyperparams
        conv_filters = hp.get('conv_filters', [32, 64, 128])
        fc_units = hp.get('fc_units', [128])
        dropout_rates = hp.get('dropout_rates', [0.5])
        # Cap dropout list to 0-2 entries
        dropout_rates = dropout_rates[:2]

        model_layers = [
            Rescaling(1.0 / 255, input_shape=input_shape),
        ]
        for f in conv_filters:
            model_layers.append(layers.Conv2D(f, (3, 3), activation='relu'))
            model_layers.append(layers.MaxPooling2D(2, 2))

        model_layers.append(layers.Flatten())

        for i, units in enumerate(fc_units):
            model_layers.append(layers.Dense(units, activation='relu'))
            if i < len(dropout_rates):
                model_layers.append(layers.Dropout(dropout_rates[i]))

        model_layers.append(layers.Dense(categories_count, activation='softmax'))
        self.model = Sequential(model_layers)

    def _compile_model(self):
        hp = self._hyperparams
        lr = float(hp.get('learning_rate', 0.001))
        try:
            opt = LegacyRMSprop(learning_rate=lr)
        except Exception:
            opt = RMSprop(learning_rate=lr)
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
