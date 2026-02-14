from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.optimizers.legacy import RMSprop as LegacyRMSprop

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential([
            Rescaling(1.0 / 255, input_shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(categories_count, activation='softmax'),
        ])

    def _compile_model(self):
        # Use legacy RMSprop on M1/M2 Macs to avoid known slowdown/crashes with v2.11+ optimizer
        try:
            opt = LegacyRMSprop(learning_rate=0.001)
        except Exception:
            opt = RMSprop(learning_rate=0.001)
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )