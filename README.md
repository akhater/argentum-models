# argentum-models

Model files that [Argentum](https://github.com/akhater/Argentum) downloads on
first use, kept here so they are never part of Argentum's own download.

Each file is attached to a release, never committed, and Argentum checks its
SHA-256 before using it.

## omnilens-v1 — `omnilens_swinir_aodlib_256.onnx`

Used by **Restore**, which undoes lens softness (a lens used wide open) at the
photo's own size.

| | |
|---|---|
| Model | OmniLens, SwinIR pretrained on the AODLib-EAOD lens library |
| Authors | Qi Jiang and colleagues, Zhejiang University |
| Paper | *OmniLens: Towards Universal Lens Aberration Correction via LensLib-to-Specific Domain Adaptation* ([arXiv 2409.05809](https://arxiv.org/abs/2409.05809)) |
| Code | [zju-jiangqi/OmniLens](https://github.com/zju-jiangqi/OmniLens), Apache-2.0 |
| Weights | [`SwinIR_AODLib_EAOD.pth`](https://huggingface.co/datasets/Strange97/OmniLens-PretrainedModel), Apache-2.0 |
| Network | SwinIR ([JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR), Apache-2.0), OmniLens's `SwinIRv2` with `upscale: 1` |
| Format | ONNX, opset 17, float32, fixed input `1×3×256×256`, sRGB-encoded RGB in 0..1 |
| Size | 53,592,179 bytes |
| SHA-256 | `c28ec5478a6db236e9151961ad1afe29ea23b80fecc8612a436f329e9a0e9630` |

### How it was made

The architecture is OmniLens's `basicsr/archs/swinir_arch.py` with its BasicSR
and timm imports replaced by plain stand-ins (`DropPath` is identity at
inference), built with the settings of its `options/test/test_SwinIR.yml`
(`embed_dim 180`, six layers of depth 6 and 6 heads, `window_size 8`,
`mlp_ratio 2`, `upsampler pixelshuffle`, `resi_connection 1conv`), loaded
strictly from `params_ema`, and exported with `torch.onnx.export(...,
opset_version=17, dynamo=False)` at a 256 px tile. The window attention fixes
the shape, so Argentum always feeds 256 px tiles with overlap.

Checked against the PyTorch original on random tiles with ONNX Runtime 1.24,
CPU and DirectML: maximum difference 0.0000.

### Licence

The ONNX file is a format conversion of the OmniLens weights and keeps their
licence, Apache-2.0 (see `LICENSE`). Credit belongs to the OmniLens and SwinIR
authors; any mistake in the conversion is Argentum's.
