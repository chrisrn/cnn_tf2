from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.nets import inception_v3, mobilenet_v2, resnet


def get_model(network, num_classes):
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


def main(args):

    data_dir = args.data_dir
    model_file = args.model_file
    network = args.network
    classes_file = args.classes_file

    if not network in ['inception_v3', 'resnet_50', 'resnet_101', 'resnet_152', 'mobilenet_v2']:
        raise ValueError('Available networks: inception_v3, resnet_50, resnet_101, resnet_152, mobilenet_v2')

    image_generator = ImageDataGenerator(rescale=1. / 255)
    image_flow = image_generator.flow_from_directory(directory=data_dir,
                                                     classes=['roses'],
                                                     class_mode=None,
                                                     target_size=(224, 224),
                                                     shuffle=False)

    with open(classes_file, 'r') as f:
        classes = f.readlines()
    # Load model
    model = get_model(network, len(classes))
    model.load_weights(model_file)

    print('Generate predictions')
    predictions = model.predict(image_flow)

    for pred in predictions:
        print(classes[np.argmax(pred)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        help='Directory of data files')
    parser.add_argument('--model_file', type=str,
                        help='Model checkpoint file')
    parser.add_argument('--network', type=str,
                        help='Network name')
    parser.add_argument('--classes_file', type=str,
                        help='Text file with classes')

    args = parser.parse_args()
    main(args)
