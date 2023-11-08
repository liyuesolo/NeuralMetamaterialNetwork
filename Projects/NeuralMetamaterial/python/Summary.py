import tensorflow as tf

class Summary():
    def __init__(self, name):
        self.summary_writer = tf.summary.create_file_writer(name)

    def saveToTensorboard(self, train_loss_gradient, train_loss_energy, validation_loss_gradient, validation_loss_energy, iter):
        with self.summary_writer.as_default():
            tf.summary.scalar('train_loss_gradient', train_loss_gradient, step=iter)
            tf.summary.scalar('train_loss_energy', train_loss_energy, step=iter)
            tf.summary.scalar('validation_loss_gradient', validation_loss_gradient, step=iter)
            tf.summary.scalar('validation_loss_energy', validation_loss_energy, step=iter)
            