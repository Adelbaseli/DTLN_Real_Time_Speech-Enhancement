"""
Export the trained DTLN checkpoint to two ONNX graphs (stage1, stage2) for
real-time frame-by-frame inference, then verify onnxruntime reproduces the
PyTorch streaming module's output.

Both graphs are exported for batch=1 (a single live audio stream) with fixed
shapes — the standard way DTLN is deployed in real time.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/best.pt --output-dir onnx
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import DTLN  # noqa: E402
from src.streaming import StreamingDTLN  # noqa: E402

STAGE1_INPUT_NAMES = ["hop_in", "ana_buf", "ola_num", "ola_den", "h1", "c1", "h2", "c2"]
STAGE1_OUTPUT_NAMES = ["out_hop", "ana_buf_out", "ola_num_out", "ola_den_out",
                        "h1_out", "c1_out", "h2_out", "c2_out"]
STAGE2_INPUT_NAMES = ["hop_in", "enc_buf", "dec_buf", "h1", "c1", "h2", "c2"]
STAGE2_OUTPUT_NAMES = ["out_hop", "enc_buf_out", "dec_buf_out",
                        "h1_out", "c1_out", "h2_out", "c2_out"]


def export_stage(module, dummy_inputs, input_names, output_names, path):
    torch.onnx.export(
        module, dummy_inputs, path,
        input_names=input_names, output_names=output_names,
        opset_version=17, dynamic_axes=None, do_constant_folding=True,
        dynamo=False,  # legacy TorchScript-based exporter: no onnxscript dependency,
                       # well-trodden path for explicit-state LSTM export
    )
    print(f"  wrote {path}")


def to_numpy(tensors):
    return [t.detach().cpu().numpy() for t in tensors]


def verify(streaming_model, stage1_path, stage2_path, n_hops=50, atol=1e-4):
    torch.manual_seed(42)
    hop = streaming_model.frame_hop
    waveform = torch.randn(1, n_hops * hop) * 0.1

    with torch.no_grad():
        torch_out = streaming_model.forward_stream(waveform)

    sess1 = ort.InferenceSession(stage1_path, providers=["CPUExecutionProvider"])
    sess2 = ort.InferenceSession(stage2_path, providers=["CPUExecutionProvider"])

    state = streaming_model.init_state(1)
    onnx_state = {
        "ana_buf": state["ana_buf"].numpy(), "ola_num": state["ola_num"].numpy(),
        "ola_den": state["ola_den"].numpy(), "enc_buf": state["enc_buf"].numpy(),
        "dec_buf": state["dec_buf"].numpy(),
        "s1_h1": to_numpy(state["s1_h1"]), "s1_h2": to_numpy(state["s1_h2"]),
        "s2_h1": to_numpy(state["s2_h1"]), "s2_h2": to_numpy(state["s2_h2"]),
    }

    onnx_chunks = []
    for i in range(n_hops):
        chunk = waveform[:, i * hop:(i + 1) * hop].numpy()

        s1_out = sess1.run(STAGE1_OUTPUT_NAMES, {
            "hop_in": chunk, "ana_buf": onnx_state["ana_buf"],
            "ola_num": onnx_state["ola_num"], "ola_den": onnx_state["ola_den"],
            "h1": onnx_state["s1_h1"][0], "c1": onnx_state["s1_h1"][1],
            "h2": onnx_state["s1_h2"][0], "c2": onnx_state["s1_h2"][1],
        })
        out_hop1, onnx_state["ana_buf"], onnx_state["ola_num"], onnx_state["ola_den"], \
            h1, c1, h2, c2 = s1_out
        onnx_state["s1_h1"], onnx_state["s1_h2"] = (h1, c1), (h2, c2)

        s2_out = sess2.run(STAGE2_OUTPUT_NAMES, {
            "hop_in": out_hop1, "enc_buf": onnx_state["enc_buf"],
            "dec_buf": onnx_state["dec_buf"],
            "h1": onnx_state["s2_h1"][0], "c1": onnx_state["s2_h1"][1],
            "h2": onnx_state["s2_h2"][0], "c2": onnx_state["s2_h2"][1],
        })
        out_hop2, onnx_state["enc_buf"], onnx_state["dec_buf"], \
            h1, c1, h2, c2 = s2_out
        onnx_state["s2_h1"], onnx_state["s2_h2"] = (h1, c1), (h2, c2)

        onnx_chunks.append(out_hop2)

    onnx_out = np.concatenate(onnx_chunks, axis=1)
    max_abs_diff = np.max(np.abs(onnx_out - torch_out.numpy()))
    print(f"  max |onnx - torch| = {max_abs_diff:.2e}")
    assert max_abs_diff < atol, (
        f"ONNX output diverges from PyTorch streaming output by {max_abs_diff:.2e} "
        f"(tolerance {atol:.0e})"
    )
    print("  ONNX output matches PyTorch streaming module.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="onnx")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    offline_model = DTLN(**ckpt["model_config"])
    offline_model.load_state_dict(ckpt["model_state_dict"])
    offline_model.eval()

    streaming_model = StreamingDTLN.from_offline(offline_model)
    streaming_model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_len, hop, hidden = (streaming_model.frame_len, streaming_model.frame_hop,
                              streaming_model.hidden_dim)
    state = streaming_model.init_state(1)

    print("Exporting stage1...")
    stage1_dummy = (
        torch.zeros(1, hop), state["ana_buf"], state["ola_num"], state["ola_den"],
        *state["s1_h1"], *state["s1_h2"],
    )
    export_stage(streaming_model.stage1, stage1_dummy, STAGE1_INPUT_NAMES,
                 STAGE1_OUTPUT_NAMES, str(out_dir / "stage1.onnx"))

    print("Exporting stage2...")
    stage2_dummy = (
        torch.zeros(1, hop), state["enc_buf"], state["dec_buf"],
        *state["s2_h1"], *state["s2_h2"],
    )
    export_stage(streaming_model.stage2, stage2_dummy, STAGE2_INPUT_NAMES,
                 STAGE2_OUTPUT_NAMES, str(out_dir / "stage2.onnx"))

    print("Verifying ONNX output vs. PyTorch streaming module...")
    verify(streaming_model, str(out_dir / "stage1.onnx"), str(out_dir / "stage2.onnx"))

    config = {
        "frame_len": frame_len, "frame_hop": hop, "hidden_dim": hidden,
        "sample_rate": args.sample_rate,
        "algorithmic_latency_samples": streaming_model.algorithmic_latency_samples,
        "source_checkpoint": str(args.checkpoint),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"  wrote {out_dir / 'config.json'}")


if __name__ == "__main__":
    sys.exit(main())
