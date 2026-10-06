# Soteria Porting Notes

Pinned official checkout: `23cf90e9e5cb41d5dc45e7540ef63a4a0ca0a8ca`.

## Mechanism

For each representation unit before the classifier, Soteria computes a sensitivity score from the norm of the derivative of that unit with respect to the input, divided by the representation magnitude (`GS_attack/reconstruct_image.py:88-103`). It selects a percentile mask and multiplies the corresponding classifier-weight gradient columns by that mask (`GS_attack/reconstruct_image.py:105-110`); this is not ordinary whole-gradient top-k pruning.

## Original configuration

The paper used `p_fc=1-40%` against DLG and `1-80%` against GS, one-sample reconstruction, L-BFGS for 300 DLG iterations, and Adam (`lr=0.1`) for 120 GS iterations. FL utility runs used SGD `lr=0.01`, local epoch 1, batch size 32, 10 of 100 clients per round; CIFAR-10 used 1,000 rounds and MNIST 200 rounds.

## Server information

The server receives an ordinary shaped update after masking and needs no secret or decoding material to average it. A white-box server knows the shared model and therefore can locate or enumerate the masked classifier layer.

## Porting

For a 13-feature MLP with BatchNorm, take the representation immediately before the final linear classifier, compute the feature-to-input Jacobian in the same model mode used to create the update, and zero its selected classifier-gradient columns. Keep BatchNorm running statistics out of the transmitted trainable-gradient vector and specify whether the attacker receives buffers.

For a roughly 137k-parameter LeNet-style CNN, use the flattened feature representation immediately before the final linear classifier and apply the same column mask. Do not silently use a convolutional activation map: the official code's explicit target is the classifier input.

## Known adaptive attacks

Balunovic et al., *Bayesian Framework for Gradient Leakage* (ICLR 2022, arXiv:2111.04706), bypass Soteria by dropping the defended layer and optimizing against remaining gradients. Yue et al., *Gradient Obfuscation Gives a False Sense of Security in Federated Learning* (USENIX Security 2023, arXiv:2206.04055), also reconstruct against Soteria with ROG.

## Reproduction status

The supplied GS stack is pinned to Linux/CUDA-era dependencies (`GS_attack/environment.yml`) and its repository invocation uses 24,000 iterations rather than the paper's 120. `../run_cpu_qualitative_checks.py` executes the official masking rule on a fixed small model; its final log is `../logs/cpu_qualitative_checks.log`. This validates mask semantics only, not paper MSE.
