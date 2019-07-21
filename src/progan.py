# TODO: make model save/load
# TODO: add dynamic levels stacking
# TODO: make config

from keras.models import Sequential, Model, Input
from keras.layers import Dense, Reshape, Conv2D, AveragePooling2D
from keras.layers import BatchNormalization, Dropout, UpSampling2D
from keras.layers import Flatten, Activation, Lambda, Add, Multiply
from keras.layers import LeakyReLU, ReLU
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.constraints import Constraint
import keras.backend as K

from tensorflow.contrib.gan import losses

import numpy as np
import os
import logging


def wass_loss(y_real, y_pred):
    return K.mean(y_real * y_pred)


def loss_discr(y_real, y_pred):
    return losses.wargs.wasserstein_discriminator_loss(y_real, y_pred)


def loss_gen(y_real, y_pred):
    return losses.wargs.wasserstein_generator_loss(y_pred)


class WeightClip(Constraint):
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__call__.__name__,
                'c': self.c}


class PROG_GAN():
    def __init__(self, img_shape_list, filter_shape_list, n_chanels, latent_shape, optimizer=Adam(0.0005, 0.5)):
        """TODO: make 'history' protected"""
        if len(filter_shape_list) != len(img_shape_list):
            raise ValueError

        self.img_shape_list = np.sort(img_shape_list)
        self.filter_shape_list = np.sort(filter_shape_list)[::-1]
        self.latent_shape = latent_shape  # flat shape

        self.n_levels = len(filter_shape_list)
        self.n_chanels = n_chanels

        self._curr_level = 0
        self._generator = None
        self._discriminator = None
        self._comb = None

        self._G_input = None
        self._G_output = None
        self._D_output = None

        self.__base = False

        self._layers_list_g = []
        self._layers_list_d = []
        self._layers_list_d_input = []

        self.optimizer = optimizer

        self.generator_loss = wass_loss
        self.discriminator_loss = wass_loss

        self.history = {
            'gen': [],
            'discr': []
        }

        logging.debug(f'filter: {self.filter_shape_list}')
        logging.debug(f'img: {self.img_shape_list}')

    def __create_g_input(self):
        """Create generator input shape"""

        n_filters = self.filter_shape_list[0]
        img_shape = self.img_shape_list[0]
        input_shape = n_filters * img_shape * img_shape

        logging.debug(f'G_input: {n_filters, img_shape}')

        model = Sequential()
        model.add(Dense(input_shape, input_shape=(self.latent_shape,)))
        model.add(ReLU())
        model.add(Reshape((img_shape, img_shape, n_filters)))
        model.name = 'GENERATOR_INPUT'
        return model

    def __create_g_output(self, layer_index):
        """Create generated image shape"""

        input_shape = (self.img_shape_list[layer_index],
                       self.img_shape_list[layer_index],
                       self.filter_shape_list[layer_index])

        logging.debug(f'G_output: {input_shape}')

        model = Sequential()
        model.add(Conv2D(self.n_chanels, kernel_size=2, padding='same', input_shape=input_shape))
        model.add(Activation('sigmoid'))
        model.name = f'GENERATOR_OUTPUT_{layer_index}'
        return model

    def __create_d_output(self):
        """Create discriminator output"""

        input_shape = (self.img_shape_list[0],
                       self.img_shape_list[0],
                       self.filter_shape_list[0])

        logging.debug(f'D_output: {input_shape}')

        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(1))
        model.name = f'DISCRIMINATOR_OUTPUT'
        return model

    def __create_g_layer(self, layer_index):
        """Create new generator level"""

        input_shape = (self.img_shape_list[layer_index],
                       self.img_shape_list[layer_index],
                       self.filter_shape_list[layer_index])

        logging.debug(f'{layer_index} layer {input_shape}')

        model = Sequential()
        model.add(Conv2D(self.filter_shape_list[layer_index], kernel_size=2, padding='same', input_shape=input_shape))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(ReLU())
        try:
            model.add(Conv2D(self.filter_shape_list[layer_index + 1], padding='same', kernel_size=2))
        except IndexError:
            model.add(Conv2D(self.filter_shape_list[layer_index], padding='same', kernel_size=2))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(ReLU())
        model.add(UpSampling2D())
        model.name = f'GENERATOR_LAYER_{layer_index}'
        return model

    def __create_d_layer(self, layer_index):
        """Create new discriminator level"""

        input_shape = (self.img_shape_list[layer_index],
                       self.img_shape_list[layer_index],
                       self.filter_shape_list[layer_index])

        logging.debug(f'{layer_index} layer {input_shape}')

        model = Sequential()
        model.add(Conv2D(self.filter_shape_list[layer_index], kernel_size=2, padding='same',
                         input_shape=input_shape, W_constraint=WeightClip(2)))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(Conv2D(self.filter_shape_list[layer_index - 1], padding='same', kernel_size=2,
                         W_constraint=WeightClip(1)))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(AveragePooling2D())
        model.name = f'DISCRIMINATOR_LAYER_{layer_index}'
        return model

    def __create_d_input(self, layer_index):
        """Create layer for image filter expanding"""

        input_shape = (self.img_shape_list[layer_index], self.img_shape_list[layer_index], self.n_chanels)

        model = Sequential()
        model.add(Conv2D(self.filter_shape_list[layer_index], kernel_size=2, padding='same',
                         input_shape=input_shape, W_constraint=WeightClip(2)))
        model.add(LeakyReLU(alpha=0.2))
        model.name = f'DISCRIMINATOR_INPUT_{layer_index}'
        return model

    def __create_layers_list(self):
        """create proggan layers stack """

        logging.debug('CREATE LAYERS LIST')
        self._layers_list_g = []
        self._layers_list_d = []
        self._layers_list_d_input = []
        self._layers_list_g_output = []
        for i in range(0, self.n_levels):
            self._layers_list_g.append(self.__create_g_layer(i))
            self._layers_list_d.append(self.__create_d_layer(i))
            self._layers_list_d_input.append(self.__create_d_input(i))
            self._layers_list_g_output.append(self.__create_g_output(i))

    def __make_trainable(self, trainable=True):
        for i in range(self.n_levels):
            self._layers_list_g[i].trainable = trainable
            self._layers_list_d[i].trainable = trainable
            self._layers_list_d_input[i].trainable = trainable
            self._layers_list_g_output[i].trainable = trainable

    def get_layer_img_shape(self, layer_index):
        return self.img_shape_list[layer_index], self.img_shape_list[layer_index], self.n_chanels

    def __build_base(self):
        if self.__base is True:
            return

        self._G_input = self.__create_g_input()
        self._D_output = self.__create_d_output()
        self.__create_layers_list()
        self.__base = True

        logging.info('initialized base model')

    def __build_generator(self, n_levels):

        alpha = Input(shape=(1, 1, 1), name='G_alpha')
        z = Input(shape=(self.latent_shape,), name='NOISE')
        x = self._G_input(z)
        for i in range(n_levels):
            x = self._layers_list_g[i](x)

        img = self._layers_list_g_output[n_levels](x)
        generator = Model(z, img)

        if n_levels > 0:
            # -3 --> previously learned model
            prev_img = generator.layers[-3].get_output_at(-1)
            prev_img = self._layers_list_g_output[n_levels - 1](prev_img)
            prev_img = UpSampling2D()(prev_img)

            prev_img = Multiply()([alpha, prev_img])
            alpha_ = Lambda(lambda x_: 1 - x_)(alpha)
            img = Multiply()([alpha_, img])

            img = Add()([prev_img, img])
            # img = Activation('sigmoid', name='SIGMOID')(img)  # not normalized?

        self._generator = Model([z, alpha], img, name='GEN')

    def __build_discriminator(self, n_levels):

        alpha = Input(shape=(1, 1, 1), name='D_alpha')
        img = Input(shape=self.get_layer_img_shape(n_levels), name='DISCR_INPUT')
        x = self._layers_list_d_input[n_levels](img)

        if n_levels > 0:
            x = self._layers_list_d[n_levels](x)
            prev_img = AveragePooling2D()(img)
            prev_img = self._layers_list_d_input[n_levels - 1](prev_img)

            prev_img = Multiply()([alpha, prev_img])
            alpha_ = Lambda(lambda x_: 1 - x_)(alpha)
            discr_img = Multiply()([alpha_, x])

            x = Add(name='COMBINATION')([prev_img, discr_img])

        for i in range(1, n_levels)[::-1]:
            x = self._layers_list_d[i](x)

        valid = self._D_output(x)
        self._discriminator = Model([img, alpha], valid, name='DISCR')
        self._discriminator.compile(self.optimizer, self.discriminator_loss)  # 'binary_crossentropy')

    def __build_combined(self):

        alpha = Input(shape=(1, 1, 1), name='comb_alpha')
        z = Input(shape=(self.latent_shape,), name='comb_z')

        self._discriminator.trainable = False
        gen_img = self._generator([z, alpha])
        valid = self._discriminator([gen_img, alpha])

        self._comb = Model([z, alpha], valid, name='COMB')
        self._comb.compile(self.optimizer, self.generator_loss)  # 'binary_crossentropy')

    def build_stacked(self, n_levels):
        self.__build_base()
        self.__make_trainable(True)

        # Generator
        self.__build_generator(n_levels)
        # Discriminator
        self.__build_discriminator(n_levels)
        # Combined
        self.__build_combined()

        # Save pic of model architecture
        self.__save_model_fig()

    def build_model(self):
        self.build_stacked(0)
        self.print_state()

    def add_level(self):
        if self.level == self.n_levels - 1:
            print(f'REACHED LAST LEVEL {self.level}')

        else:
            self._curr_level += 1
            self.build_stacked(self.level)
        self.print_state()

    def train(self, real_X, alpha=0):
        self.train_discriminator(real_X, alpha)
        self.train_generator(len(real_X), alpha)

    def train_discriminator(self, real_X, alpha=0):
        batch_size = len(real_X)
        noise = np.random.normal(0, 1, (batch_size, self.latent_shape))
        alpha = alpha * np.ones((real_X.shape[0], 1, 1, 1))

        gen_X = self.generate(noise, alpha)
        y_real = np.ones(batch_size)
        y_gen = np.zeros(batch_size) - 1
        loss_real = self._discriminator.train_on_batch([real_X, alpha], y_real)
        loss_gen = self._discriminator.train_on_batch([gen_X, alpha], y_gen)
        self.history['discr'].append((loss_real + loss_gen) / 2)

    def train_generator(self, batch_size, alpha=0):
        noise = np.random.normal(0, 1, (batch_size, self.latent_shape))
        alpha = alpha * np.ones((batch_size, 1, 1, 1))
        loss = self._comb.train_on_batch([noise, alpha], np.ones(batch_size))
        self.history['gen'].append(loss)

    def generate(self, noise=None, alpha=0):
        if noise is None:
            noise = np.random.normal(0, 1, (1, self.latent_shape))
        alpha = alpha * np.ones((len(noise), 1, 1, 1))
        return self._generator.predict([noise, alpha])

    def get_proba(self, X, alpha=0):
        alpha = alpha * np.ones((len(X), 1, 1, 1))
        return self._discriminator.predict([X, alpha])

    def get_comb_proba(self, noise=None, alpha=0):
        img = self.generate(noise, alpha)
        return self.get_proba(img, alpha)

    def __save_model_fig(self, save_to='../data/logs/'):
        for model in (self._generator, self._discriminator, self._comb):
            path = os.path.join(save_to, model.name + '.png')
            plot_model(model, to_file=path, show_shapes=True)

    def print_state(self):
        print(f'Number of stacked levels: {self.level}')
        print(f'Image shape: {self.get_layer_img_shape(self.level)}')

    def save_model(self, path):

        # MAKE FOLDER FOR MODELS SAVING
        # SAVE MODEL PARAMS TO PICKLE OR TEXT FILE

        # SAVE THIS TO MODEL FOLDER
        self._G_input
        self._D_output

        # SAVE EACH ELEMENT TO MODEL FOLDER
        self._layers_list_g
        self._layers_list_d
        self._layers_list_d_input

        # FOLDER -> ARCHIVE

        pass

    def load_model(self):

        # TAKE ARCHIVE PATH, OPEN
        # LOAD PARAMS
        # LOAD LEVELS

        # BUILD MODEL ACCORDING TO CURRENT PARAMS
        pass

    @property
    def output_shape(self):
        return self._generator.output_shape

    @property
    def level(self):
        return self._curr_level

# if __name__ == '__main__':
#     IMG_SHAPE_LIST = [8, 16, 32, 64, 128, 256]
#     FILTER_SHAPE_LIST = [16, 32, 64, 128, 256, 512]
#     PROG_GAN(IMG_SHAPE_LIST, FILTER_SHAPE_LIST, 3, (100,))._PROG_GAN__create_G_input()
