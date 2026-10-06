"""Read-only import isolation for project models versus TabLeak's generic names."""
import sys

# P34 tabular models and P30 image imports share the generic name `models`.
# Temporarily isolate external generic packages while importing immutable image
# references; restore the project namespace/path before any training begins.
prefixes=('models','attacks','datasets','utils')
saved={name:module for name,module in sys.modules.items()
       if any(name==prefix or name.startswith(prefix+'.') for prefix in prefixes)}
old_path=list(sys.path)
for name in saved:
    del sys.modules[name]
try:
    from experiments import priority31_image_utility_dp as p31
    from experiments import priority33c_image_recovery as recovery
finally:
    for name in list(sys.modules):
        if any(name==prefix or name.startswith(prefix+'.') for prefix in prefixes):
            del sys.modules[name]
    sys.modules.update(saved)
    sys.path[:]=old_path
