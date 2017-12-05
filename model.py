from __future__ import division
import os
import time
import math
from glob import glob
import scipy.io as sio
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
    def __init__(
            self,
            sess,
            input_height=450,
            input_width=450,
            crop=True,
            batch_size=64,
            sample_num=64,
            output_height=128,
            output_width=128,
            y_dim=1,
            z_dim=256,
            gf_dim=64,
            df_dim=64,
            gfc_dim=1024,
            dfc_dim=1024,
            c_dim=3,
            dataset_name='default',
            input_fname_pattern='*.jpg',
            checkpoint_dir=None,
            sample_dir=None):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient
        # flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')
        self.d_bn5 = batch_norm(name='d_bn5')
        self.d_bn6 = batch_norm(name='d_bn6')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')
        self.g_bn5 = batch_norm(name='g_bn5')
        self.g_bn6 = batch_norm(name='g_bn6')
        self.g_bn7 = batch_norm(name='g_bn7')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        self.data = glob(os.path.join(
            "./data", self.dataset_name, self.input_fname_pattern))
        self.data.sort()

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(self.data)

        imreadImg = imread(self.data[0])
        # check if image is a non-grayscale image by checking channel
        # number
        self.data_y = self.load_labels()
        if len(imreadImg.shape) >= 3:
            self.c_dim = imread(self.data[0]).shape[-1]
        else:
            self.c_dim = 1

        self.grayscale = (self.c_dim == 1)

        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(
                tf.float32, [self.batch_size, self.y_dim], name='y')
        else:
            self.y = None

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')

        inputs = self.inputs

        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.G = self.generator(self.z, self.y)
        self.D, self.D_logits, self.fD = self.discriminator(
            inputs, self.y, reuse=False)
        self.sampler = self.sampler(self.z, self.y)
        self.D_, self.D_logits_, self.fD_ = self.discriminator(
            self.G, self.y, reuse=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=x, labels=y)
            except BaseException:
                return tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(
                self.D_logits, tf.ones_like(
                    self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(
                self.D_logits_, tf.zeros_like(
                    self.D_)))
        # d_loss
        self.d_loss = self.d_loss_real + self.d_loss_fake

        # g_loss
        self.g_loss = tf.reduce_mean(
            self.D_logits) - tf.reduce_mean(self.D_logits_)

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        d_optim = tf.train.AdamOptimizer(
            config.learning_rate,
            beta1=config.beta1) .minimize(
            self.d_loss,
            var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(
            config.learning_rate,
            beta1=config.beta1) .minimize(
            self.g_loss,
            var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except BaseException:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                    self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        sample_files = self.data[0:self.sample_num]
        sample = [
            get_image(
                sample_file,
                input_height=self.input_height,
                input_width=self.input_width,
                resize_height=self.output_height,
                resize_width=self.output_width,
                crop=self.crop,
                grayscale=self.grayscale) for sample_file in sample_files]
        if (self.grayscale):
            sample_inputs = np.array(sample).astype(
                np.float32)[:, :, :, None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)
        sample_labels = self.data_y[0:self.sample_num]

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            self.data = glob(os.path.join(
                "./data", config.dataset, self.input_fname_pattern))
            self.data.sort()

            seed = 547
            np.random.seed(seed)
            np.random.shuffle(self.data)
            
            batch_idxs = min(
                len(self.data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = self.data[idx *
                                        config.batch_size:(idx +
                                                           1) *
                                        config.batch_size]
                batch = [
                    get_image(
                        batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for batch_file in batch_files]
                if self.grayscale:
                    batch_images = np.array(batch).astype(
                        np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)
                batch_labels = self.data_y[idx *
                                           config.batch_size:(idx +
                                                              1) *
                                           config.batch_size]

                batch_z = np.random.uniform(-1,
                                            1,
                                            [config.batch_size,
                                             self.z_dim]) .astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={
                    self.inputs: batch_images, self.z: batch_z, self.y: batch_labels})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={
                                               self.inputs: batch_images, self.z: batch_z, self.y: batch_labels})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to
                # zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={
                                               self.inputs: batch_images, self.z: batch_z, self.y: batch_labels})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval(
                    {self.z: batch_z, self.y: batch_labels})
                errD_real = self.d_loss_real.eval(
                    {self.inputs: batch_images, self.y: batch_labels})
                errG = self.g_loss.eval(
                    {self.inputs: batch_images, self.z: batch_z, self.y: batch_labels})

                counter += 1
                print(
                    "Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" %
                    (epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 100) == 1:
                    try:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                                self.y: sample_labels,
                            },
                        )
                        save_images(
                            samples, image_manifold_size(
                                samples.shape[0]), './{}/train_{:02d}_{:04d}.png'.format(
                                    config.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" %
                              (d_loss, g_loss))
                    except BaseException:
                        print("one pic error!...")
                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)
            h0 = lrelu(conv2d(x, 64, 5, 5, 2, 2, name='d_h0_conv'))
            h0 = conv_cond_concat(h0, yb)

            h1 = lrelu(self.d_bn1(
                conv2d(h0, 92, 5, 5, 2, 2, name='d_h1_conv')))
            h1 = conv_cond_concat(h1, yb)

            h2 = lrelu(self.d_bn2(
                conv2d(h1, 128, 5, 5, 2, 2, name='d_h2_conv')))
            h2 = conv_cond_concat(h2, yb)

            h3 = lrelu(self.d_bn3(
                conv2d(h2, 256, 5, 5, 2, 2, name='d_h3_conv')))
            h3 = conv_cond_concat(h3, yb)

            h4 = lrelu(self.d_bn4(
                conv2d(h3, 256, 3, 3, 2, 2, name='d_h4_conv')))
            h4 = conv_cond_concat(h4, yb)

            h5 = lrelu(self.d_bn5(
                conv2d(h4, 256 * 4 * 4, 3, 3, 2, 2, name='d_h5_conv')))
            h5 = tf.reshape(
                h5, [self.batch_size, -1])
            h5 = concat([h5, y], 1)

            h6 = linear(h5, 1, 'd_h6_lin')

            return tf.nn.sigmoid(h6), h6, h5

    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)
            # 2 fully-connected layers
            h0 = lrelu(
                self.g_bn0(
                    linear(
                        z,
                        self.gfc_dim,
                        'g_h0_lin')))
            h0 = concat([h0, y], 1)

            h1 = lrelu(
                self.g_bn1(
                    linear(
                        h0,
                        256 * 8 * 8,
                        'g_h1_lin')))
            h1 = tf.reshape(h1, [self.batch_size, 2, 2, 256 * 4 * 4])
            h1 = conv_cond_concat(h1, yb)

            # 6 deconv layers with 2-by-2 upsampling
            h2 = lrelu(
                self.g_bn2(
                    deconv2d(
                        h1, [
                            self.batch_size, 4, 4, 256], 3, 3, 2, 2, name='g_h2')))
            h2 = conv_cond_concat(h2, yb)

            h3 = lrelu(
                self.g_bn3(
                    deconv2d(
                        h2, [
                            self.batch_size, 8, 8, 256], 3, 3, 2, 2, name='g_h3')))
            h3 = conv_cond_concat(h3, yb)

            h4 = lrelu(
                self.g_bn4(
                    deconv2d(
                        h3, [
                            self.batch_size, 16, 16, 128], 3, 3, 2, 2, name='g_h4')))
            h4 = conv_cond_concat(h4, yb)

            h5 = lrelu(
                self.g_bn5(
                    deconv2d(
                        h4, [
                            self.batch_size, 32, 32, 92], 5, 5, 2, 2, name='g_h5')))
            h5 = conv_cond_concat(h5, yb)

            h6 = lrelu(
                self.g_bn6(
                    deconv2d(
                        h5, [
                            self.batch_size, 64, 64, 64], 5, 5, 2, 2, name='g_h6')))
            h6 = conv_cond_concat(h6, yb)

            h7 = tf.nn.tanh(
                self.g_bn7(
                    deconv2d(
                        h6, [
                            self.batch_size, 128, 128, 3], 5, 5, 2, 2, name='g_h7')))

            return h7

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)
            # 2 fully-connected layers
            h0 = lrelu(
                self.g_bn0(
                    linear(
                        z,
                        self.gfc_dim,
                        'g_h0_lin')))
            h0 = concat([h0, y], 1)

            h1 = lrelu(
                self.g_bn1(
                    linear(
                        h0,
                        256 * 8 * 8,
                        'g_h1_lin')))
            h1 = tf.reshape(h1, [self.batch_size, 2, 2, 256 * 4 * 4])
            h1 = conv_cond_concat(h1, yb)

            # 6 deconv layers with 2-by-2 upsampling
            h2 = lrelu(
                self.g_bn2(
                    deconv2d(
                        h1, [
                            self.batch_size, 4, 4, 256], 3, 3, 2, 2, name='g_h2')))
            h2 = conv_cond_concat(h2, yb)

            h3 = lrelu(
                self.g_bn3(
                    deconv2d(
                        h2, [
                            self.batch_size, 8, 8, 256], 3, 3, 2, 2, name='g_h3')))
            h3 = conv_cond_concat(h3, yb)

            h4 = lrelu(
                self.g_bn4(
                    deconv2d(
                        h3, [
                            self.batch_size, 16, 16, 128], 3, 3, 2, 2, name='g_h4')))
            h4 = conv_cond_concat(h4, yb)

            h5 = lrelu(
                self.g_bn5(
                    deconv2d(
                        h4, [
                            self.batch_size, 32, 32, 92], 5, 5, 2, 2, name='g_h5')))
            h5 = conv_cond_concat(h5, yb)

            h6 = lrelu(
                self.g_bn6(
                    deconv2d(
                        h5, [
                            self.batch_size, 64, 64, 64], 5, 5, 2, 2, name='g_h6')))
            h6 = conv_cond_concat(h6, yb)

            h7 = tf.nn.tanh(
                self.g_bn7(
                    deconv2d(
                        h6, [
                            self.batch_size, 128, 128, 3], 5, 5, 2, 2, name='g_h7')))

            return h7

    def load_labels(self):
        labels = sio.loadmat('imagelabels.mat')
        labels = labels['labels']
        labels = labels.T

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(labels)

        labels = labels/51 - 1
        labels = np.asarray(labels)

        return labels

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
