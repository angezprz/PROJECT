from tensorflow.keras import layers, models, optimizers

def build_lstm_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=input_shape),
        layers.LSTM(32),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model
