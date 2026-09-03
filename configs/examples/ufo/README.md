# UFO-Deploy examples

These three policies use the released G1 29-DoF artifacts in `../../UFO` when
`vex-policy` is started from this repository root. Model and context paths may
instead be any absolute or working-directory-relative local paths.

```bash
uv run vex-policy --policy-config configs/examples/ufo --interface eth0
```

Selecting a UFO policy first interpolates from the activation pose to UFO's
default standing pose for 10 seconds. Policy inference runs during this period
to warm its four-frame histories. Tracking then plays once from `start_frame`
through `end_frame` and holds the latent at `stop_frame`.

Set `task.startup_mode: prefill` to skip interpolation. The startup state and
its equivalent residual action fill all four history frames immediately.
`startup_mode` only accepts `prefill` and `interpolate`; `init_duration_s` only
applies to `interpolate`.

The independent top-level `guard` checks that the robot is upright and close
to UFO's default pose before either startup mode runs. Its
`startup_joint_tolerance_rad` and `startup_gravity_tolerance` thresholds must
pass; otherwise activation enters the normal Vex latch path without writing a
UFO command. The check is implemented by `UfoGuard`, following the waist
locomotion guard structure.

Reward and goal examples intentionally bind one named latent to one Vex policy.
Duplicate a YAML and change `name`, context `name`, and reward `z_id` to expose
another latent. Use the released `reward_locomotion_numpy.pkl`; the older
`reward_locomotion.pkl` needs Torch and is not part of this runtime.

Invalid state, observations, actions, or inference results latch the Vex state
machine. Send a non-estop command with an empty policy list before selecting a
policy again. Context files are pickle/joblib data and must come from a trusted
source.
