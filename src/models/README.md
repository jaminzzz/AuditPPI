# Model package

This package is the reusable downstream-model layer for AuditPPI.

```text
models/
├── architectures/   Project-owned torch.nn.Module definitions
│   ├── mlp_pair.py
│   ├── mlp_endpoint.py
│   └── tabm_pair.py
└── estimators/      External libraries with fit/predict-style APIs
    ├── xgboost.py
    ├── tabpfn.py
    └── ebm.py
```

Import concrete implementations directly:

```python
from src.models.architectures.mlp_endpoint import MLPEndpoint
from src.models.architectures.mlp_pair import train_mlp_pair
from src.models.architectures.tabm_pair import train_tabm_pair
from src.models.estimators.xgboost import fit_xgb, fit_xgb_classifier
```

`src.ppi_fingerprint.baseline` imports these implementations directly.
Baseline-specific networks such as DeepNano are intentionally not part of this
package.
