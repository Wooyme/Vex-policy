# SONIC model files

Copy the three upstream ONNX files here before loading a SONIC configuration:

- `gear_sonic_deploy/policy/release/model_decoder.onnx` → `model_decoder.onnx`
- `gear_sonic_deploy/policy/release/model_encoder.onnx` → `model_encoder.onnx`
- `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx` → `planner_sonic.onnx`

The weights are intentionally excluded from Git because the planner model is
approximately 739 MB. Paths can instead be changed in each policy YAML.
