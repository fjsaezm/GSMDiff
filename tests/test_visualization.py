import matplotlib
import pytest
import torch

from gsmdiff.visualization import (
    draw_high_pass_contours,
    high_pass_coefficients,
    high_pass_magnitude,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def test_high_pass_response_detects_a_vertical_step():
    image = torch.zeros(1, 8, 8)
    image[:, :, 4:] = 1.0
    response = high_pass_magnitude(image)
    assert response.shape == (8, 8)
    assert torch.all(response[:, 3] == 1.0)
    assert torch.count_nonzero(response) == 8


def test_signed_coefficients_pool_both_filter_directions():
    image = torch.zeros(1, 3, 4)
    image[:, :, 2:] = 1.0
    coefficients = high_pass_coefficients(image)
    expected_count = 1 * (3 * 3 + 2 * 4)
    assert coefficients.shape == (expected_count,)
    assert torch.count_nonzero(coefficients) == 3
    assert torch.all(coefficients[coefficients != 0] == -1.0)


def test_contour_overlay_draws_the_thresholded_response():
    image = torch.zeros(1, 8, 8)
    image[:, :, 4:] = 1.0
    figure, axis = plt.subplots()
    axis.imshow(image[0], cmap="gray")
    mask = draw_high_pass_contours(axis, image, threshold=0.5)
    assert torch.equal(mask, high_pass_magnitude(image) > 0.5)
    assert len(axis.images) == 2
    plt.close(figure)


def test_contour_parameters_are_validated():
    figure, axis = plt.subplots()
    with pytest.raises(ValueError, match="quantile"):
        draw_high_pass_contours(axis, torch.zeros(4, 4), quantile=1.1)
    plt.close(figure)
