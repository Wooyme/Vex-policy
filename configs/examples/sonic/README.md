# GEAR-SONIC modes

This directory contains one complete policy YAML for each of the 27 planner
modes. It is nested intentionally, so the existing default `../../g1`
launch remains unchanged. Run all SONIC modes with:

```bash
uv run vex-policy --policy-config configs/g1/sonic
```

All configurations use the same decoder, encoder, and planner sessions; the
runtime cache loads each ONNX model only once.
