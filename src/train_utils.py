import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.use('TkAgg')

import os
import shutil
import numpy as np

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.utils import multi_gpu_model

from src.nets import inception_v3, mobilenet_v2, resnet


# Class for learning rate view during training
class LRlogs(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        """
        Prints the learning rate during training
        :param epoch: <int> num epoch
        :param logs: <str> for stdouts
        """
        lr = tf.keras.backend.eval(self.model.optimizer.lr)
        print('learning_rate: {}'.format(lr))


class ModelHandler(object):
    def __init__(self,
                 callbacks,
                 gpus):
        """
        Model parameters initialization
        Args:
            callbacks: <dict> with callbacks we want to activate
            gpus: <int> number of gpus to utilize
        """

        # Training callbacks

        # Adaptive Learning Rate Decay
        self.adaptive_lr = callbacks['adaptive_learning_rate']
        self.adaptive_lr_patience_epochs = callbacks['adaptive_lr_patience_epochs']
        self.adaptive_lr_decay = callbacks['adaptive_lr_decay']
        self.min_adaptive_lr = callbacks['min_adaptive_lr']

        # Early Stopping
        self.early_stopping = callbacks['early_stopping']
        self.early_stopping_min_change = callbacks['early_stopping_min_change']
        self.early_stopping_patience_epochs = callbacks['early_stopping_patience_epochs']

        # Exponential Learning Rate Decay
        self.exponential_lr = callbacks['exponential_lr']
        self.num_epochs_per_decay = callbacks['num_epochs_per_decay']
        self.lr_decay_factor = callbacks['lr_decay_factor']

        # Tensorboard
        self.logdir = callbacks['logdir']
        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir)

        # Model checkpoint
        self.model_dir = callbacks['model_dir']
        self.save_per_epoch = callbacks['save_per_epoch']
        if os.path.exists(self.model_dir) and self.save_per_epoch:
            shutil.rmtree(self.model_dir)

        self.gpus = gpus

    def train_test_model(self, hparams, train_generator, val_generator):
        """
        Tf.keras model construction and fit
        Args:
            hparams: <dict> with parameters of training
            train_generator: <tf generator> for train set
            val_generator: <tf generator> for test set

        Returns: <dict> of training history,
                 <numpy array> of predictions on test set,
                 <dict> of metrics on test set

        """
        num_classes = train_generator.num_classes
        model = self.get_model(hparams['network'], num_classes)
        if self.gpus:
            model = multi_gpu_model(model, self.gpus)

        print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

        optimizer = self.get_optimizer(hparams['optimizer'])(hparams['learning_rate'])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics='accuracy')

        # model.summary()
        callbacks = self.get_callbacks()

        steps_per_epoch = train_generator.n // hparams['batch_size']
        validation_steps = val_generator.n // hparams['batch_size']
        history = model.fit(train_generator,
                            epochs=hparams['epochs'],
                            validation_data=val_generator,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=validation_steps,
                            validation_freq=1,
                            callbacks=callbacks)

        # Generate predictions (probabilities -- the output of the last layer)
        # on new data using `predict`

        print('Generate predictions')
        predictions = model.predict(val_generator)
        print('predictions shape:', predictions.shape)
        test_metrics = model.evaluate(val_generator, return_dict=True)
        return history.history, predictions, test_metrics

    def get_model(self, network, num_classes):
        """

        Args:
            network: <str> network name
            num_classes: <int> number of output classes

        Returns:

        """

        if network == 'inception_v3':
            model = inception_v3.InceptionV3(include_top=True,
                                             weights=None,
                                             input_tensor=None,
                                             input_shape=None,
                                             pooling=None,
                                             classes=num_classes,
                                             classifier_activation='softmax')
        elif network == 'resnet_50':
            model = resnet.ResNet50(include_top=True,
                                    weights=None,
                                    input_tensor=None,
                                    input_shape=None,
                                    pooling=None,
                                    classes=num_classes,
                                    classifier_activation='softmax')
        elif network == 'resnet_101':
            model = resnet.ResNet101(include_top=True,
                                     weights=None,
                                     input_tensor=None,
                                     input_shape=None,
                                     pooling=None,
                                     classes=num_classes,
                                     classifier_activation='softmax')
        elif network == 'resnet_152':
            model = resnet.ResNet152(include_top=True,
                                     weights=None,
                                     input_tensor=None,
                                     input_shape=None,
                                     pooling=None,
                                     classes=num_classes,
                                     classifier_activation='softmax')
        else:
            base_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                  alpha=1.0,
                                                  include_top=False,
                                                  weights='imagenet',
                                                  input_tensor=None,
                                                  pooling=None,
                                                  classifier_activation='softmax')
            base_model.trainable = False
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
        return model

    def get_callbacks(self):
        """
        Callbacks initialization
        :return: <list> with activated callbacks
        """
        callbacks = []

        if self.adaptive_lr:
            print('***** Adaptive Learning Rate callback activated *****')
            patience = self.adaptive_lr_patience_epochs
            factor = self.adaptive_lr_decay
            min_lr = self.min_adaptive_lr
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=factor,
                                                             patience=patience, min_lr=min_lr)
            callbacks.append(reduce_lr)

        if self.early_stopping:
            print('***** Early Stopping callback activated *****')
            min_delta = self.early_stopping_min_change
            patience = self.early_stopping_patience_epochs
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta,
                                                              patience=patience, verbose=0,
                                                              mode='min', baseline=None,
                                                              restore_best_weights=False)
            callbacks.append(early_stopping)

        if self.save_per_epoch:
            print('***** Model checkpoint callback activated *****')
            filepath = os.path.join(self.model_dir, "cp-{epoch:04d}.ckpt")
            ckpt = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                      monitor='val_acc',
                                                      verbose=1,
                                                      save_weights_only=True,
                                                      save_freq='epoch')
            callbacks.append(ckpt)

        if self.exponential_lr:
            print('***** Exponential Learning Rate decay activated *****')

            def schedule(epoch, lr):
                epochs_per_decay = self.num_epochs_per_decay
                decay_factor = self.lr_decay_factor
                if epoch % epochs_per_decay == 0 and epoch != 0:
                    return lr * decay_factor
                else:
                    return lr
            exp_lr = tf.keras.callbacks.LearningRateScheduler(schedule)
            callbacks.append(exp_lr)

        callbacks.append(LRlogs())

        return callbacks

    def get_optimizer(self, name):
        """
        Getting optimizer
        Args:
            name: <str> optimizer name

        Returns: <tf.keras> optimizer

        """

        if name == "adam":
            return tf.keras.optimizers.Adam
        elif name == "sgd":
            return tf.keras.optimizers.SGD
        elif name == "adagrad":
            return tf.keras.optimizers.Adagrad
        elif name == "adadelta":
            return tf.keras.optimizers.Adadelta
        elif name == "rmsprop":
            return tf.keras.optimizers.RMSprop
        else:
            raise ValueError("You should feed for optimizer a value between: adam, sgd, adagrad, adadelta, rmsprop")


def get_train_params(args):
    """
    Hparams objects' construction
    Args:
        args: <dict> of train parameters

    Returns: <dict> of hparams train objects

    """
    hparams = []

    # Batch size
    batch_size = args['batch_size']
    batch_size = hp.HParam('batch_size', hp.Discrete(batch_size))
    hparams.append(batch_size)

    # Num epochs
    epochs = args['epochs']
    epochs = hp.HParam('epochs', hp.Discrete(epochs))
    hparams.append(epochs)

    # Learning rate
    lr = args['learning_rate']
    lr = hp.HParam('learning_rate', hp.Discrete(lr))
    hparams.append(lr)

    # Optimizer
    optimizer = args['optimizer']
    if not any(opt in ['adam', 'sgd', 'adagrad', 'adadelta', 'rmsprop'] for opt in optimizer):
        raise ValueError('Available optimizers: adam, sgd, adagrad, adadelta, rmsprop')
    optimizer = hp.HParam('optimizer', hp.Discrete(optimizer))
    hparams.append(optimizer)

    # Activation function in each layer
    activation = args['activation']
    if not any(act in ['relu', 'sigmoid', 'tanh'] for act in activation):
        raise ValueError('Available activations: relu, sigmoid or tanh')
    activation = hp.HParam('activation', hp.Discrete(activation))
    hparams.append(activation)

    # Network
    network = args['network']
    if not any(nn in ['inception_v3', 'resnet_50', 'resnet_101', 'resnet_152', 'mobilenet_v2'] for nn in network):
        raise ValueError('Available networks: inception_v3, resnet_50, resnet_101, resnet_152, mobilenet_v2')
    network = hp.HParam('network', hp.Discrete(network))
    hparams.append(network)

    # Pre-processing
    preprocessing = args['preprocessing']
    if not any(pp in ['inception', 'resnet', 'mobilenet'] for pp in preprocessing):
        raise ValueError('Available preprocessings: inception, resnet, mobilenet')
    preprocessing = hp.HParam('preprocessing', hp.Discrete(preprocessing))
    hparams.append(preprocessing)

    params = {'batch_size': batch_size,
              'epochs': epochs,
              'learning_rate': lr,
              'optimizer': optimizer,
              'activation': activation,
              'network': network,
              'preprocessing': preprocessing}

    return params


def plot_metric(best_train, best_val, metric='accuracy'):
    """
    Plot ground-truth vs predicted values
    Args:
        y_test: <numpy> array of test labels
        best_predictions: <numpy> array of predictions
        test_time: <pandas data-frame> column of test dates
        best_loss: <float> loss


    """

    plt.plot(best_train, label='Training')
    plt.plot(best_val, label='Validation')
    plt.title('{}: {}'.format(metric, best_val[-1]))
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.close()

