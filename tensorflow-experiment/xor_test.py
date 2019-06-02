import numpy as np
import pandas as pd
import tensorflow as tf


def test_xor():
    training_data = pd.DataFrame([
        {'a': 0, 'b': 0, 'result': 0},
        {'a': 0, 'b': 1, 'result': 1},
        {'a': 1, 'b': 0, 'result': 1},
        {'a': 1, 'b': 1, 'result': 0}
    ], dtype='float32')

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, input_dim=2, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(
        optimizer=tf.train.GradientDescentOptimizer(3),
        loss=tf.keras.losses.mean_squared_error,
        metrics=['binary_accuracy'])

    history = model.fit(
        x=training_data[['a', 'b']].values, 
        y=training_data[['result']].values, 
        epochs=1000)

    print(model.summary())
    print(model.get_weights())
    
    predictions = model.predict(training_data[['a', 'b']]) >= 0.5
    assert predictions[:, 0].tolist() == [False, True, True, False]
