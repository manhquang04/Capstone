# FetchSGD Count-Sketch Porting Notes

Pinned official checkout: `833ca44cc43a9b034515f55485524e4f1d0fad21`.

## Mechanism

Each worker creates `CSVec`, accumulates its flattened update, and uploads the sketch table (`CommEfficient/fed_worker.py:311-320`). The server sums tables (`fed_aggregator.py:320-332`), inserts the result into `CSVec` (`fed_aggregator.py:568-585`), and uses `unSketch(k)` (`fed_aggregator.py:590`) before re-sketching residual error (`:593-601`).

## Original configuration

The defaults are `k=50000`, five rows, 500,000 columns, 20 blocks, seed 21 (`CommEfficient/utils.py:107-111,141-147`). FetchSGD reports communication and convergence, not a reconstruction/privacy metric; therefore it has no published headline privacy number or DP/noise comparator.

## Server information

Correct linear aggregation and decoding require clients and server to use identical bucket and sign hashes. The checkout independently constructs `CSVec` without an explicit hash/seed argument (`fed_worker.py:311-316`; `fed_aggregator.py:464-467`) and does not vendor or pin `csvec`; a future port must make this material explicit, and must state whether the server knows it. In the intended FetchSGD protocol, the server receives the aggregate sketch and decodes approximate top-k aggregate coordinates.

## Porting

For either the 13-feature BatchNorm MLP or LeNet-style CNN, concatenate all trainable parameter gradients in a registered order, sketch with shared hash functions, sum tables across clients, then unsketch top-k at the server. Preserve an index/shape manifest to map the sparse decoded vector back into tensors; exclude BatchNorm buffers unless the FL protocol normally synchronizes them.

## Known attacks

Song et al., *Sketching for First Order Method: Efficient Algorithm for Low-Bandwidth Channel and Vulnerability* (ICML 2023, arXiv:2210.08371), give a reconstruction attack for shared random-sketch aggregation when the attacker knows the sketch matrix. It is not a verified reproduction against FetchSGD's exact `CSVec` Count-Sketch implementation; see `../REPRODUCTION_REPORT.md` for the scoped search result.

## Reproduction status

The official stack imports undeclared external `csvec`, its test is stale, and full training unconditionally initializes CUDA/NCCL. `../run_cpu_qualitative_checks.py` verifies fixed-hash Count-Sketch linearity (maximum float32 addition error `2.861e-06`) and shows aggregation fails with mismatched hashes (maximum discrepancy `19.9291`); see `../logs/cpu_qualitative_checks.log`.
