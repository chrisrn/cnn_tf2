import pytest
import pandas as pd
import numpy as np
import json

from src.data_utils import DataHandler


def get_data_handler():
    # Get params to create data object
    params_file = '../src/params.json'
    with open(params_file) as json_file:
        params = json.load(json_file)

    # Object to test data
    data_handler = DataHandler(params['data'], params['train']['batch_size'])
    return data_handler


@pytest.fixture(scope='class')
def data_constructor(request):
    # Data-frames for distance and cartesians computation
    request.cls.distance_row = pd.DataFrame({'source_latitude': [-12.088156],
                                             'source_longitude': [-77.016065],
                                             'destination_latitude': [-12.108531],
                                             'destination_longitude': [-77.044891]}).iloc[0]

    request.cls.cartesian_row = pd.DataFrame({'source_latitude': [-12.088156],
                                              'source_longitude': [-77.016065]}).iloc[0]

    # Approximate radius of earth in km
    request.cls.R = 6373.0

    # Data handler object
    request.cls.data_handler = get_data_handler()


@pytest.mark.usefixtures('data_constructor')
class TestData:
    def test_distance(self):
        distance = self.data_handler.add_distance(self.distance_row)

        expected_distance = 3.86846
        assert distance == pytest.approx(expected_distance, 0.00001)

    def test_x_cartesian(self):
        x_coordinate = self.data_handler.add_x(self.cartesian_row, 'source', self.R)

        expected_x = 1400.1224
        assert x_coordinate == pytest.approx(expected_x, 0.0001)

    def test_y_cartesian(self):
        y_coordinate = self.data_handler.add_y(self.cartesian_row, 'source', self.R)

        expected_y = -6072.3636
        assert y_coordinate == pytest.approx(expected_y, 0.0001)

    def test_z_cartesian(self):
        z_coordinate = self.data_handler.add_z(self.cartesian_row, 'source', self.R)

        expected_z = -1334.6109
        assert z_coordinate == pytest.approx(expected_z, 0.0001)
