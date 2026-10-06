# Priority34C — model and adapter implementation preflight

Before synthetic integration tests or scientific targets. Ratio uses the exact
unchanged models/fraud_mlp.py loaded under an isolated module name, initialization
seed340042, eval mode, focal alpha.95/gamma2, CPU. Its input is the P32 prepared
vector (all13/476/58 coordinates). Exclude every BN affine parameter and buffer
from transmitted gradient; retain fixed public initial BN in the known model.
Ratio scoring converts prepared vectors into the train-standardized scoring
adapter, not a new model preprocessing rule. Scoring and prior controls use
unchanged P33B mixed-feature accuracy, tolerances and bounds.

TabLeak retains public initialization333042 and unchanged official config46
through P33B native_recover. Native representations reuse the byte-identical
P33B IEEE60/BAF58 adapter documents; PaySim uses all13 prepared coordinates,
with train-only mean/sample-standard-deviation, bounds and logical blocks as
P33B. Full IEEE ratio adapter retains every logical block, not the selected60.
No official or earlier source is edited. Namespace-isolated artifact writes
may target only artifacts/priority34c.

Defenses retain P33B frozen v1 seed681958327/conservative settings and v2
seed20260916/.95/.01. V2 ratio receives sketches and known metadata and uses
the frozen least-squares preflight; native TabLeak uses its original sketch
space matching. Local epsilon10 noise is applied to each individual gradient:
global clip.01, SD.08031273039270019, independent private Gaussian generator.
Noise generators must not perturb training/attack RNG. Never hand raw gradients
or realized noise to the attacker. This preflight does not draw targets or
authorize scientific execution; final source firewall and execution freeze
remain required.
