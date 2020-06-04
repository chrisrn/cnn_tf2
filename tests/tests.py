import pytest
import json
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.train_utils import ModelHandler


@pytest.fixture(scope='class')
def constructor(request):

    with open('params_test.json') as json_file:
        params = json.load(json_file)
    request.cls.params = params

    # Get generator for toy data
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    train_dir = params['data']['data_dir']
    train_image_generator = ImageDataGenerator(rescale=1. / 255)
    request.cls.train_data_gen = train_image_generator.flow_from_directory(batch_size=2,
                                                                           directory=train_dir,
                                                                           shuffle=True,
                                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                           class_mode='categorical')

    # Get model
    request.cls.model_handler = ModelHandler(params['callbacks'],
                                             params['more']['gpus'])


@pytest.mark.usefixtures('constructor')
class TestData:
    def train_one_step(self):
        params = self.params['hyper_parameters']
        num_classes = self.train_data_gen.num_classes

        model = self.model_handler.get_model(params['network'][0], num_classes)
        optimizer = self.model_handler.get_optimizer(params['optimizer'][0])(params['learning_rate'][0])

        weights_before = model.get_weights()
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics='accuracy')
        history = model.fit(self.train_data_gen,
                            steps_per_epoch=1)
        weights_after = model.get_weights()

        all_vars = [var.name for var in model.variables]
        trainable_vars = [var.name for var in model.trainable_variables]

        return history.history, all_vars, trainable_vars, weights_before, weights_after

    def test_weights_change(self):

        _, all_vars, trainable_vars, weights_before, weights_after = self.train_one_step()
        # Make sure trainable weights are changed
        for b, a, v in zip(weights_before, weights_after, all_vars):
            if v in trainable_vars:
                assert np.any(np.not_equal(b, a))

    def test_loss_nonzero(self):
        history, _, _, _, _ = self.train_one_step()
        loss_one_step = history['loss'][0]
        assert loss_one_step != 0



