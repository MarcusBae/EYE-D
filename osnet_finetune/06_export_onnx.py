"""06_export_onnx.py — 파인튜닝된 osnet_x0_25 를 ONNX로 추출

EYE-D 서버 사양과 동일하게 맞춤:
    - 모델  : osnet_x0_25
    - 입력  : (N, 3, 256, 128)  RGB, ImageNet normalize 는 호출 측에서 처리 가정
    - 출력  : (N, 512), L2 normalized

사용:
    python 06_export_onnx.py \
        --weights log/osnet_x0_25_eyed/model/model.pth.tar-60 \
        --out exports/osnet_x0_25.onnx \
        --image-size 256 128 --l2-norm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OSNetWrapper(nn.Module):
    """OSNet feature extractor + (optional) L2 normalization."""

    def __init__(self, backbone: nn.Module, l2: bool = True):
        super().__init__()
        self.backbone = backbone
        self.l2 = l2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torchreid 모델은 eval 모드에서 feature 벡터를 반환
        feat = self.backbone(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        if self.l2:
            feat = F.normalize(feat, p=2, dim=1)
        return feat


def load_model(weights: str, name: str = "osnet_x0_25") -> nn.Module:
    from torchreid import models
    from torchreid.utils import load_pretrained_weights

    # num_classes 는 export 시점에는 무관 (feature 만 추출)
    model = models.build_model(name=name, num_classes=1000, pretrained=False)
    load_pretrained_weights(model, weights)
    model.eval()
    return model


def verify(onnx_path: Path, ref_model: nn.Module, image_size, l2_norm: bool):
    import onnxruntime as ort

    H, W = image_size
    dummy = torch.randn(2, 3, H, W)
    with torch.no_grad():
        ref = ref_model(dummy)
        if isinstance(ref, (list, tuple)):
            ref = ref[0]
        if l2_norm:
            ref = F.normalize(ref, p=2, dim=1)
    ref_np = ref.numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})[0]

    diff = np.max(np.abs(ref_np - out))
    print(f"PyTorch vs ONNX  max abs diff = {diff:.6e}")
    print(f"output shape: {out.shape}  (expected (N, 512))")
    if l2_norm:
        norms = np.linalg.norm(out, axis=1)
        print(f"L2 norms (should be ~1.0): {norms}")
    assert diff < 1e-3, "출력 차이가 큽니다. 가중치/전처리 확인 필요"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help=".pth.tar 경로")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="osnet_x0_25")
    p.add_argument("--image-size", nargs=2, type=int, default=[256, 128],
                   metavar=("H", "W"))
    p.add_argument("--l2-norm", action="store_true", default=True)
    p.add_argument("--opset", type=int, default=14)
    p.add_argument("--dynamic-batch", action="store_true", default=True)
    p.add_argument("--simplify", action="store_true", default=True)
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading weights: {args.weights}")
    backbone = load_model(args.weights, args.name)
    model = OSNetWrapper(backbone, l2=args.l2_norm).eval()

    H, W = args.image_size
    dummy = torch.randn(1, 3, H, W)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    torch.onnx.export(
        model, dummy, str(out),
        input_names=["input"], output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"exported: {out}")

    if args.simplify:
        try:
            import onnx
            import onnxsim
            m = onnx.load(str(out))
            m_simp, ok = onnxsim.simplify(m)
            if ok:
                onnx.save(m_simp, str(out))
                print("onnx-simplifier 적용 완료")
            else:
                print("[warn] onnx-simplifier 실패 (계속 진행)")
        except Exception as e:
            print(f"[warn] simplify skipped: {e}")

    print("verifying ...")
    verify(out, model, args.image_size, args.l2_norm)
    print("done.")


if __name__ == "__main__":
    main()
