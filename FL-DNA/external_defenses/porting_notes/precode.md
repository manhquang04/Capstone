# PRECODE Porting Notes

Pinned official checkout: `c66adc4cdd62993139eafebc1b57fc25b0694874`.

## Mechanism

The variational bottleneck encodes a feature vector into `mu` and `softplus(std)`, samples `mu + eps*std`, then decodes it (`src/VariationalBottleneck.py:31-32,47-60,72-74`). Training adds its beta-weighted KL penalty to cross entropy (`src/VariationalBottleneck.py:76-77`; `src/training.py:42-49`).

## Original configuration

The paper inserts one module before the final classifier with latent dimension `k=256` and `beta=0.001`. It trained 300 epochs with Adam `lr=0.001`, betas `(0.9, 0.999)`, batch size 64; its IGA attack used Gaussian dummy initialization, cosine loss, TV `1e-6`, Adam `lr=0.01`, at most 7,000 iterations and a 1,200-iteration no-improvement stop.

## Server information

No key or decoder is held by the server: the bottleneck is part of the shared model and aggregation is ordinary FedAvg. A white-box server can know all bottleneck weights and architecture; the only client-side randomness is the particular reparameterization sample used during update creation.

## Porting

For a 13-feature MLP with BatchNorm, place the module after the final hidden/BatchNorm activation and immediately before the classifier. Keep its stochastic sample enabled while computing a transmitted training update; use fixed client RNG capture only if an experiment explicitly gives it to the attacker.

For a roughly 137k-parameter LeNet-style CNN, flatten the last feature layer and insert the bottleneck before the final linear classifier, matching the paper's placement. If using a spatial tensor, the official module has a reducer branch (`src/VariationalBottleneck.py:24-32`), but flattening the final feature vector is the closest paper analogue.

## Known adaptive attacks

Scheliga et al., *Dropout is NOT All You Need to Prevent Gradient Leakage* (arXiv:2208.06163), optimize stochastic masks; Scheliga et al., *Privacy Preserving Federated Learning with Convolutional Variational Bottlenecks* (arXiv:2309.04515), show an attack that omits stochastic gradients. Yue et al. (USENIX Security 2023, arXiv:2206.04055) also breaks PRECODE with ROG.

## Reproduction status

The supplied notebook depends on an uninitialized `invertinggradients` submodule and a Linux/CUDA environment. `../run_cpu_qualitative_checks.py` imports the official bottleneck and confirms two fixed-weight forward/backward passes differ due to sampling; see `../logs/cpu_qualitative_checks.log`. This is not the 128-image, 7,000-step paper inversion evaluation.
