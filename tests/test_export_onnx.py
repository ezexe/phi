"""export_onnx with the packages the export extra declares, on the torch installed."""

import numpy as np
import pytest
import torch
import torch.nn as nn

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


def test_export_onnx_writes_a_model_onnxruntime_runs(tmp_path):
    from phi_prune.api import prune
    from phi_prune.export import export_onnx

    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10))
    pruned, masks = prune(model)
    path = str(tmp_path / "model.onnx")
    export_onnx(pruned, path)
    onnx.checker.check_model(onnx.load(path))
    # A batch of 2 runs through the batch axis the export names dynamic
    x = torch.randn(2, 3, 32, 32)
    session = ort.InferenceSession(path)
    out = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
    assert np.allclose(out, pruned(x).detach().numpy(), atol=1e-5)
