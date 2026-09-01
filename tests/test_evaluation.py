import pytest
import torch

from gsmdiff.scripts.evaluate_fid_kid import _source_count, _tensor_to_uint8


def test_fid_tensor_conversion_maps_model_range_to_uint8():
    images = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
    converted = _tensor_to_uint8(images, "minus_one_one")
    assert converted.dtype == torch.uint8
    assert converted.shape == (1, 3, 1, 3)
    assert converted[0, 0].tolist() == [[0, 128, 255]]


def test_fid_source_count_reads_multiple_tensor_batches(tmp_path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    torch.save(torch.zeros(3, 3, 2, 2), first)
    torch.save({"samples": torch.zeros(5, 3, 2, 2)}, second)
    assert _source_count([first, second]) == 8


def test_fid_tensor_conversion_rejects_ambiguous_float_range():
    with pytest.raises(ValueError, match="Cannot infer"):
        _tensor_to_uint8(torch.full((1, 3, 2, 2), 2.0), "auto")
