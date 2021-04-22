import tensorflow as tf

class LinearRegression():
    def __init__(self, lr=0.01, num_epochs=100, batch_size=10):
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.w = None
        self.b = None

    def __h(self, x, w, b):
        return tf.tensordot(x, w, axes=1) + b

    def __mse_deriv(self, y_true, y_pred):
        return tf.reshape(tf.reduce_mean(2 * (y_pred - y_true)), [1, 1])

    def fit(self, x, y):
        x = tf.constant(x, dtype=tf.float32)
        y = tf.constant(y, dtype=tf.float32)

        num_samples, num_features = x.shape

        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(num_samples).repeat(self.num_epochs).batch(self.batch_size)

        iterator = dataset.__iter__()

        self.w = tf.zeros([num_features, 1])
        self.b = 0

        for _ in range(self.num_epochs):
            for _ in range(int(num_samples / self.batch_size)):
                x_batch, y_batch = iterator.get_next()

                y_pred = self.__h(x_batch, self.w, self.b)

                dJ_dH = self.__mse_deriv(y_batch, y_pred)
                dH_dW = x_batch
                dJ_dW = tf.reduce_mean(dJ_dH * dH_dW)
                dJ_dB = tf.reduce_mean(dJ_dH)

                self.w -= self.lr * dJ_dW
                self.b -= self.lr * dJ_dB

    def predict(self, x):
        x = tf.constant(x, dtype=tf.float32)
        return self.__h(x, self.w, self.b)
