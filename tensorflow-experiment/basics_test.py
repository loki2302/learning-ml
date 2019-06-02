import tensorflow as tf


def test_variables():
    g = tf.Graph()
    with g.as_default():
        x = tf.Variable(3, name='x')
        y = tf.add(x, 2, name='y')

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            session.run(x.assign(111))
            x_value = session.run(x)
            y_value = session.run(y)
            assert x_value == 111
            assert y_value == 113

            session.run(x.assign(1))
            x_value = session.run(x)
            y_value = session.run(y)
            assert x_value == 1
            assert y_value == 3


def test_placeholders():
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(tf.float32)
        y = tf.add(x, 2, name='y')

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            y_value = session.run(y, feed_dict={x: 1})
            assert y_value == 3

            y_value = session.run(y, feed_dict={x: 2})
            assert y_value == 4
