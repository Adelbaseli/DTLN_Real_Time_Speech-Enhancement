"""
Benchmark real-time factor (RTF) and per-frame latency of the exported ONNX
streaming model on CPU with onnxruntime — the actual number that matters for
"can this run in real time on modest hardware".

RTF = (wall-clock time to process one hop) / (duration of audio in one hop).
RTF < 1 means the model processes audio faster than it arrives, i.e. it can
run live without falling behind.

Usage:
    python scripts/benchmark_rtf.py --onnx-dir onnx [--frame-hop 128] [--seconds 30]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

STAGE1_OUTPUT_NAMES = ["out_hop", "ana_buf_out", "ola_num_out", "ola_den_out",
                        "h1_out", "c1_out", "h2_out", "c2_out"]
STAGE2_OUTPUT_NAMES = ["out_hop", "enc_buf_out", "dec_buf_out",
                        "h1_out", "c1_out", "h2_out", "c2_out"]


def zeros_like_input(sess, name):
    for inp in sess.get_inputs():
        if inp.name == name:
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            return np.zeros(shape, dtype=np.float32)
    raise KeyError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", default="onnx")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-hop", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=30.0,
                         help="How much synthetic audio to stream through the benchmark.")
    parser.add_argument("--warmup-hops", type=int, default=50)
    parser.add_argument("--output", default="results/benchmark.md")
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # single-thread = worst-case/most representative of edge hardware
    sess1 = ort.InferenceSession(str(onnx_dir / "stage1.onnx"), sess_options=opts,
                                  providers=["CPUExecutionProvider"])
    sess2 = ort.InferenceSession(str(onnx_dir / "stage2.onnx"), sess_options=opts,
                                  providers=["CPUExecutionProvider"])

    state = {name: zeros_like_input(sess1, name) for name in
              ["ana_buf", "ola_num", "ola_den", "h1", "c1", "h2", "c2"]}
    s1_state = {"ana_buf": state["ana_buf"], "ola_num": state["ola_num"],
                "ola_den": state["ola_den"], "h1": state["h1"], "c1": state["c1"],
                "h2": state["h2"], "c2": state["c2"]}
    s2_state = {name: zeros_like_input(sess2, name) for name in
                ["enc_buf", "dec_buf", "h1", "c1", "h2", "c2"]}

    hop = args.frame_hop
    n_hops = int(args.seconds * args.sample_rate / hop)
    rng = np.random.default_rng(0)
    hops = rng.standard_normal((n_hops, 1, hop)).astype(np.float32) * 0.1

    latencies_ms = []
    for i in range(n_hops):
        t0 = time.perf_counter()

        out1 = sess1.run(STAGE1_OUTPUT_NAMES, {"hop_in": hops[i], **s1_state})
        out_hop1, s1_state["ana_buf"], s1_state["ola_num"], s1_state["ola_den"], \
            s1_state["h1"], s1_state["c1"], s1_state["h2"], s1_state["c2"] = out1

        out2 = sess2.run(STAGE2_OUTPUT_NAMES, {"hop_in": out_hop1, **s2_state})
        _, s2_state["enc_buf"], s2_state["dec_buf"], \
            s2_state["h1"], s2_state["c1"], s2_state["h2"], s2_state["c2"] = out2

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if i >= args.warmup_hops:  # drop JIT/cache warm-up hops from the stats
            latencies_ms.append(elapsed_ms)

    latencies_ms = np.array(latencies_ms)
    hop_duration_ms = 1000 * hop / args.sample_rate
    mean_latency = latencies_ms.mean()
    p95_latency = np.percentile(latencies_ms, 95)
    rtf = mean_latency / hop_duration_ms

    report = (
        f"# Real-time factor benchmark (CPU, single-thread onnxruntime)\n\n"
        f"- Frames benchmarked: {len(latencies_ms)} (after {args.warmup_hops} warm-up hops)\n"
        f"- Hop size: {hop} samples ({hop_duration_ms:.2f} ms of audio @ {args.sample_rate}Hz)\n"
        f"- Mean per-hop processing latency: {mean_latency:.3f} ms\n"
        f"- p95 per-hop processing latency: {p95_latency:.3f} ms\n"
        f"- **RTF: {rtf:.3f}** ({'real-time capable' if rtf < 1 else 'NOT real-time capable'} "
        f"— RTF < 1 means processing is faster than audio arrives)\n"
    )
    print(report)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    sys.exit(main())
