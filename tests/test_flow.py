import pytest
import torch

from gsmdiff.flow import PnPFlowUNet, PretrainedPnPFlowModel


def test_pnp_flow_checkpoint_adapter_loads_strict_state_dict(tmp_path):
    architecture = {
        "input_channels": 3,
        "input_height": 8,
        "channels": 32,
        "channel_multipliers": (1, 2),
        "residual_blocks": 1,
        "attention_resolutions": (),
    }
    source = PnPFlowUNet(
        input_channels=architecture["input_channels"],
        input_height=architecture["input_height"],
        ch=architecture["channels"],
        ch_mult=architecture["channel_multipliers"],
        num_res_blocks=architecture["residual_blocks"],
        attn_resolutions=architecture["attention_resolutions"],
    ).eval()
    checkpoint = tmp_path / "flow.pt"
    torch.save(source.state_dict(), checkpoint)
    loaded = PretrainedPnPFlowModel.from_pretrained(
        checkpoint,
        device=torch.device("cpu"),
        dtype=torch.float32,
        **architecture,
    )
    sample = torch.randn(2, 3, 8, 8)
    time = torch.tensor(0.4)
    with torch.no_grad():
        expected = source(sample, time.repeat(2))
    assert torch.equal(loaded.velocity(sample, time), expected)
    assert loaded.sample_size == 8
    assert loaded.in_channels == 3


def test_pnp_flow_loader_explains_how_to_enable_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.download=true"):
        PretrainedPnPFlowModel.from_pretrained(
            tmp_path / "missing.pt",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_pnp_flow_loader_rejects_wrong_checkpoint_checksum(tmp_path):
    checkpoint = tmp_path / "untrusted.pt"
    checkpoint.write_bytes(b"not the configured checkpoint")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        PretrainedPnPFlowModel.from_pretrained(
            checkpoint,
            device=torch.device("cpu"),
            dtype=torch.float32,
            sha256="0" * 64,
        )
