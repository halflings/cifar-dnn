import os
import string

import numpy as np
import tensorflow as tf


class CNN:

    def __init__(self, image_size, image_channels, batch_size, learning_rate, batch_norm,
                 conv_layers, fc_units, log_root='/tmp/cnn-logs'):
        self.image_size = image_size
        self.image_channels = image_channels
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.conv_layers = conv_layers
        self.fc_units = fc_units
        self.log_dir = self.get_new_summary_dir(log_root)

        self.build_model()

    def build_model(self):
        self.session = tf.Session()
        self.image_batch = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.image_size, self.image_size, self.image_channels],
            name='image_batch')
        self.label_batch = tf.placeholder(dtype=tf.int32, shape=[self.batch_size],
                                          name='label_batch')
        # CONV -> POOL -> CONV -> POOL -> CONV -> POOL -> DENSE
        cur_input = self.image_batch
        for filters, kernel_size in self.conv_layers:
            cur_input = tf.layers.conv2d(cur_input, filters=8, kernel_size=3,
                                         padding='same', activation=tf.nn.relu)
            cur_input = tf.layers.max_pooling2d(
                cur_input, pool_size=2, strides=[1, 1], padding='same')
            if self.batch_norm:
                cur_input = tf.layers.batch_normalization(cur_input)
        self.logits = tf.layers.dense(tf.reshape(
            cur_input, [self.batch_size, -1]), units=self.fc_units, activation=tf.nn.relu)
        self.predictions = tf.argmax(self.logits, axis=1)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.label_batch, self.logits)
        self.train_op = optimizer.minimize(self.loss)

        # Summary
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('logits_argmax', self.predictions)
        tf.summary.scalar('loss', self.loss)
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.session.graph)
        self.summaries = tf.summary.merge_all()

        # Misc
        self.step = tf.Variable(1, name='step', trainable=False, dtype=tf.int32)
        self.increment_step_op = tf.assign(self.step, self.step + 1)

    def train(self, images, labels, max_epochs, log_period=20):
        self.session.run(tf.global_variables_initializer())
        for epoch in range(max_epochs):
            indices = np.random.permutation(len(images))[:self.batch_size]
            feed_dict = {self.image_batch: images[indices], self.label_batch: labels[indices]}
            _, _, cur_loss = self.session.run(
                [self.train_op, self.increment_step_op, self.loss], feed_dict=feed_dict)
            if epoch % log_period == 0:
                print("Epoch #{}/{} : loss = {:.4f}".format(epoch, max_epochs, cur_loss))
                summary = self.session.run(self.summaries, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary, self.step.eval(self.session))

    def get_new_summary_dir(self, summary_root):
        if not os.path.exists(summary_root):
            os.makedirs(summary_root)
        num_dirs = [int(d.split('_')[0]) for d in os.listdir(summary_root)
                    if all(c in string.digits for c in d.split('_')[0])]
        next_dir_num = max(num_dirs) + 1 if num_dirs else 0
        next_dir = '{}_batch_norm={}-conv_layers={}-fc={}-lr={}-batch_size={}'.format(
            next_dir_num, self.batch_norm, self.conv_layers, self.fc_units, self.learning_rate,
            self.batch_size)
        next_dir_path = os.path.join(summary_root, next_dir)
        os.makedirs(next_dir_path)
        return next_dir_path


if __name__ == '__main__':
    import cifar
    images, labels, label_names = cifar.load_dataset()
    cnn = CNN(image_size=32, image_channels=3, batch_size=64, learning_rate=0.0005,
              batch_norm=True, conv_layers=[(6, 3), (18, 3), (32, 3)], fc_units=600)
    cnn.train(images, labels, max_epochs=700, log_period=10)
