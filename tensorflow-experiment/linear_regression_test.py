import tensorflow as tf
import numpy as np
import math


def test_linear_regression():
    x = np.linspace(0, 50, 50)
    y = 2 * x + 3

    g = tf.Graph()
    with g.as_default():
        X = tf.placeholder(tf.float32)
        Y = tf.placeholder(tf.float32)
        W = tf.Variable(np.random.randn(), name='W')
        b = tf.Variable(np.random.randn(), name='b')

        y_pred = tf.add(tf.multiply(X, W), b)

        cost = tf.reduce_sum(tf.pow(y_pred - Y, 2)) / (2 * len(x))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03).minimize(cost)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for iteration in range(1, 1000):
                for (_x, _y) in zip(x, y):
                    session.run(optimizer, feed_dict={
                        X: _x,
                        Y: _y
                    })

                if iteration % 100 == 0:
                    cost_value = session.run(cost, feed_dict={
                        X: x,
                        Y: y
                    })
                    weight = session.run(W)
                    bias = session.run(b)
                    print(f'iteration {iteration}, cost={cost_value}, W={weight}, b={bias}')

            cost_value = session.run(cost, feed_dict={X: x, Y: y})
            weight = session.run(W)
            bias = session.run(b)
            print(f'finally, cost={cost_value}, W={weight}, b={bias}')

            assert math.isclose(cost_value, 0, abs_tol=0.001)
            assert math.isclose(weight, 2.0, abs_tol=0.1)
            assert math.isclose(bias, 3.0, abs_tol=0.1)
