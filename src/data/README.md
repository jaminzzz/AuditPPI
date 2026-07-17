# Data package

`src.data` defines reusable input data contracts and low-level readers. It does
not contain feature construction, model fitting, metrics, or experiment output
loading.

```text
data/
├── benchmarks.py   Benchmark object + C1/C2/C3, cross-species, RF2-PPI loaders
├── sequences.py    Sequence normalization, stable IDs, and FASTA readers
└── sae_cache.py    Per-residue sparse SAE LMDB codec and reader
```

Typical imports:

```python
from src.data import Benchmark, load_benchmark
from src.data.sequences import read_fasta
from src.data.sae_cache import SaeCacheReader
```

Evaluation belongs in `src.eval`, while protein and pair feature construction
belongs in `src.features`.
