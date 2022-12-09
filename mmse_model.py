

L2 = 0.01

mmse_model = keras.models.Sequential(
    [
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(L2)),
        keras.layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(L2)),
        keras.layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(L2)),
        keras.layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(L2)),
        keras.layers.Dense(1),

    ]
)