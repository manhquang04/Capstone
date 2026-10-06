# Priority32b read-only activation-path diagnosis

FROZEN before this probe, after the fixed backend probe completed. Original
training design, runner and previous probe artifacts remain unchanged. The
backend probe found five of six unchanged registered final checkpoints produce
0/1024 finite CPU outputs but 1024/1024 finite MPS outputs on identical input;
the isolated BN operator produces NaN for negative variance on BOTH backends.
This warrants locating where the full-model NaN disappears, not changing scores.

For ALL six registered final checkpoints and the same first1024 prepared P32
validation records, read-only forward hooks capture shape/NaN/infinity counts at
every sequential layer, once CPU and once MPS. Never alter outputs, parameters
or buffers. Also evaluate fixed torch.relu([NaN,-1,0,1]) on each backend to test
activation NaN propagation directly. No training, clamping, replacement,
calibration, seed selection or extra hypothesis tests. Threads1 and restored CPU
constructor RNG. Write only new activation_probe artifacts; distinguish observed
operator behavior from inferred consequences for utility. These diagnostic
inputs do not supply utility scores or quantify a valid-BN counterfactual bias.
