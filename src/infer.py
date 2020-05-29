from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

from train_utils import plot_predictions

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def numpy_features(df):
    """
    Keeping only the features we want for training
    Args:
        df: <pandas data-frame>

    Returns: <numpy arrays> of data

    """
    df = df.reset_index(drop=True)

    data = df['requests']
    data.index = df['request_date']

    data = data.values

    return data


def rescale(df):
    """
    Rescale values for lstm input
    Args:
        df: <pandas data-frame>

    Returns: <pandas data-frame> rescaled, <scaler object>

    """
    scaler = MinMaxScaler()
    train_features = list(df.columns)
    train_features.remove('request_date')
    for feature in train_features:
        df[feature] = scaler.fit_transform(df[[feature]])

    return df, scaler


def main(args):
    start_date = args.start_date.replace(',', ' ')
    end_date = args.end_date.replace(',', ' ')

    # Read data
    df = pd.read_csv(args.data_file, sep='\t')

    df, scaler = rescale(df)

    df = df[['request_date', 'requests']]

    # Ground-truth data-frame
    gt_df = df[df['request_date'] >= start_date]
    gt_df = gt_df[gt_df['request_date'] <= end_date]
    gt_df.reset_index(inplace=True, drop=True)

    # Get first sequence of input data
    input_data = df[df['request_date'] < start_date]
    input_data = input_data[-args.history_size:]

    input_data = numpy_features(input_data)
    input_data = np.reshape(input_data, newshape=(1, -1, 1))

    # Load model
    model = tf.keras.models.load_model(args.model_file)

    predictions = []
    ground_truths = []
    avg_mse = 0
    for i in range(0, len(gt_df), args.target_size):

        prediction_scaled = model.predict(input_data)

        diff = len(gt_df) - i
        if diff < args.target_size:
            rows = gt_df.iloc[i:]
            prediction_scaled = prediction_scaled[0, :diff]
            prediction_scaled = np.expand_dims(prediction_scaled, 0)
        else:
            rows = gt_df.iloc[i: i + args.target_size]

        requests = rows['requests'].values
        # MSE for this prediction
        mse = mean_squared_error(requests, prediction_scaled[0])
        avg_mse += mse

        # Get real values of prediction and ground-truth
        prediction = scaler.inverse_transform(prediction_scaled)
        ground_truth = scaler.inverse_transform([requests])
        ground_truths.append(ground_truth)
        predictions.append(prediction)

        # LIFO for new input
        input_data = np.delete(input_data, list(range(0, args.target_size)))
        input_data = np.insert(input_data, len(input_data), prediction_scaled[0])
        input_data = np.reshape(input_data, newshape=(1, -1, 1))

    mse = avg_mse / len(gt_df)
    ground_truths = np.squeeze(np.hstack(ground_truths))
    predictions = np.squeeze(np.hstack(predictions))

    plot_predictions(ground_truths, predictions, gt_df['request_date'], mse)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str,
                        help='Csv file of data on which the model is trained')
    parser.add_argument('--model_file', type=str,
                        help='Model checkpoint h5 file')
    parser.add_argument('--start_date', type=str,
                        help='Start date of predictions')
    parser.add_argument('--end_date', type=str,
                        help='End date of predictions')
    parser.add_argument('--history_size', type=int,
                        help='Time-series history window')
    parser.add_argument('--target_size', type=int,
                        help='Time-series target window')

    args = parser.parse_args()
    main(args)
