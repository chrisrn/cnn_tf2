from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import itertools
import numpy as np
import tensorflow as tf

from data_utils import DataHandler
from train_utils import ModelHandler, get_train_params
from tensorboard.plugins.hparams import api as hp

from train_utils import plot_metric


def main(params_file):
    """
    Main function
    Args:
        params_file: <str> json file with parameters


    """
    with open(params_file) as json_file:
        params = json.load(json_file)

    # Object to handle data
    data_handler = DataHandler(params['data'])

    # Object to handle training and evaluation processes
    model_handler = ModelHandler(params['callbacks'],
                                 params['more']['gpus'])

    # Get hyper-parameters and construct experiments
    train_params = get_train_params(params['hyper_parameters'])
    param_names = params['hyper_parameters'].keys()
    lists = [train_params[name].domain.values for name in param_names]
    experiments = list(itertools.product(*lists))

    # Run hyper-parameter tuning experiments
    best_hparams = {}
    session_num = 0
    best_acc = 0
    best_train_acc = []
    best_val_acc = []
    best_train_loss = []
    best_val_loss = []
    for index, experiment in enumerate(experiments):
        hparams = {name: experiment[i] for i, name in enumerate(param_names)}

        run_name = "run-{}".format(session_num)
        print('--- Starting trial: {}'.format(run_name))
        print(hparams)
        run_dir = params['callbacks']['logdir'] + '/hparam_tuning/' + run_name
        train_generator, val_generator = data_handler.load_data(hparams['batch_size'], hparams['preprocessing'])
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            history, predictions, test_metrics = model_handler.train_test_model(hparams,
                                                                                train_generator,
                                                                                val_generator)
            train_acc = history['accuracy']
            val_acc = history['val_accuracy']
            train_loss = history['loss']
            val_loss = history['val_loss']
            for epoch in range(hparams['epochs']):
                tf.summary.scalar('Training Accuracy', train_acc[epoch], step=epoch)
                tf.summary.scalar('Validation Accuracy', val_acc[epoch], step=epoch)
                tf.summary.scalar('Training Loss', train_loss[epoch], step=epoch)
                tf.summary.scalar('Validation Loss', val_loss[epoch], step=epoch)
        if test_metrics['accuracy'] > best_acc:
            best_train_acc = train_acc
            best_val_acc = val_acc
            best_train_loss = train_loss
            best_val_loss = val_loss
            best_acc = test_metrics['accuracy']
            best_hparams = hparams.copy()
            best_predictions = predictions
        session_num += 1

    print('Best experiment with test accuracy {}'.format(best_acc))
    print(best_hparams)
    plot_metric(best_train_acc, best_val_acc)
    plot_metric(best_train_loss, best_val_loss, 'categorical cross entropy')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, default='params.json',
                        help='Json file with parameters for training and data')

    args = parser.parse_args()
    main(args.params)
