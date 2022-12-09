

initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=200,
    decay_rate=0.96,
    staircase=True)


loss_tracker = keras.metrics.Mean()
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)