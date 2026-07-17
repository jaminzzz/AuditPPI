# Model package

This package is the reusable downstream-model layer for AuditPPI.

```text
models/
├── architectures/   Project-owned torch.nn.Module definitions
│   ├── dual_tower.py
│   ├── endpoint_mlp.py
│   └── tabm_pair.py
├── estimators/      External libraries with fit/predict-style APIs
│   ├── xgboost.py
│   ├── tabpfn.py
│   └── ebm.py
└── registry.py      Lazy name-to-implementation resolution
```

Use explicit imports when a concrete implementation is known:

```python
from src.models.architectures.endpoint_mlp import EndpointMLP
from src.models.estimators.xgboost import fit_xgb, fit_xgb_classifier
```

Use the registry for configuration-driven experiments:

```python
from src.models import build_architecture, get_estimator_factory

model = build_architecture("dual_tower", feat_dim=1280, fuse="sym")
fit_xgb = get_estimator_factory("xgboost")
```

`src.ppi_fingerprint.baseline` imports these implementations directly.
Baseline-specific networks such as DeepNano are intentionally not part of this
package.
