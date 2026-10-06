# Phase 3 follow-up: prospective development and confirmation

1. Analyze existing reconstruction errors by feature, class, client and restart.
   Existing artifacts remain untouched.
2. Development uses the complete original client 2 (73,862 records). Raw attack
   and zero-update follow one trajectory each to 500, saving best-so-far at
   100/300/500. Restart-0 initialization matches the previous run exactly.
3. Compare single-change candidates at 100 steps with the raw 100-step
   reference: public loss rescaling by local learning rate squared (1e-6), and
   softmax categorical relaxation alone, and balance-difference parameterization
   alone (raw-unit equations followed by fixed scaling). Local training is unchanged.
   Softmax is a continuous simplex representation, not valid hard one-hot.
   Each candidate has its own equally parameterized unoptimized prior and a
   paired zero-update control. No claim of strongest no-update control is made.
   The balance candidate was added during development after two candidate results
   were available, before freeze/confirmation; see development_amendment.json.
4. Select the smallest baseline-minus-own-prior mean MSE on development at 100
   steps. Freeze before constructing confirmation targets. Confirmation has
   100 iterations and two restarts per client; this is the tested candidate
   budget, not a claim of convergence. The 500-step reference is diagnostic.
5. Draw a fresh stratified 500k subset excluding every source row in the original
   500k subset. Use its 65% training portion for three new client sets with the
   same partition algorithm. Retain the old train-fitted scaler and checkpoint.
   Save source IDs, assert disjointness, record seeds and dataset hash. All local
   records participate, using Adam, batch 1024 and one epoch. Labels/order/RNG
   remain privileged attacker knowledge. Native replay gates each attack.

Report mean/median MSE, class/feature errors, categorical behavior, candidate
reload, pairing and timing. Selection uses development reconstruction errors;
candidate selection inside an attack uses only its objective. Confirmation
errors must not change configurations or budgets. No formal privacy claim.

Closure can be positive, negative or inconclusive. A numerical success threshold
is not invented; weak benefit versus prior cannot justify ranking DNA. This
follow-up tests one selected configuration, not exhaustive attack optimality.
