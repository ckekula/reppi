# reppi

[![GitHub](https://img.shields.io/badge/repo-GitHub-black?logo=github)](https://github.com/ckekula/reppi)
[![Docs](https://img.shields.io/badge/docs-online-blue?logo=readthedocs)](https://ckekula.github.io/reppi/reppi.html)

A Python library for representation learning — sparse coding and dictionary learning algorithms implemented close to their original formulations.

## Installation

```bash
pip install reppi
```

## Algorithms

| Algorithm | Class | Reference |
|-----------|-------|-----------|
| Orthogonal Matching Pursuit | `OMP` | Elad et al., 2008 |
| K-SVD | `KSVD` | Aharon et al., 2006 |
| Label Consistent K-SVD | `LCKSVD` | Jiang et al., 2011 |
| Frozen Dictionary Learning | `IncrementalFrozenDictionary` | Carroll et al., 2017 |

## Convention

reppi follows the **column-major convention** common in the sparse representation literature: signals are columns, so data matrices are shaped `(n_features, n_samples)`. This matches the MATLAB toolboxes the implementations are based on. If your data is in sklearn's `(n_samples, n_features)` layout, transpose before passing it in.

## Quick Start

### Installaion

```bash
pip install reppi
```

### Sparse Coding with OMP

```python
from reppi import OMP
import numpy as np

# D: (n_features, n_atoms), unit-norm columns
# X: (n_features, n_samples)
omp = OMP(n_nonzero_coefs=10)
Gamma = omp.encode(X, D)  # (n_atoms, n_samples)
```

### Dictionary Learning with K-SVD

```python
from reppi import KSVD

ksvd = KSVD(
    n_components=128,     # number of atoms
    n_nonzero_coefs=10,   # sparsity level T
    n_iter=20,
    verbose=True,
)
ksvd.fit(X_train)         # X_train: (n_features, n_samples)

D = ksvd.D_               # learned dictionary (n_features, n_components)
Gamma = ksvd.transform(X) # sparse codes (n_components, n_samples)
```

### Discriminative Dictionary Learning with LC-KSVD

LC-KSVD jointly learns a dictionary and a linear classifier from labelled data. Labels are passed as a one-hot matrix `H` of shape `(n_classes, n_samples)`.

**LC-KSVD1** — reconstruction + label-consistency:

```python
from reppi import LCKSVD

model = LCKSVD(
    n_components=570,
    n_nonzero_coefs=30,
    alpha=4.0,            # weight for label-consistency term
    variant="lcksvd1",
    n_iter=50,
    n_iter_init=20,       # K-SVD iterations for initialisation
    verbose=True,
)
model.fit(X_train, H_train)
Gamma = model.transform(X_test)  # (n_components, n_samples)
```

**LC-KSVD2** — reconstruction + label-consistency + classification error:

```python
model = LCKSVD(
    n_components=570,
    n_nonzero_coefs=30,
    alpha=4.0,
    beta=2.0,             # weight for classifier term
    variant="lcksvd2",
    n_iter=50,
    n_iter_init=20,
    verbose=True,
)
model.fit(X_train, H_train)

predictions = model.predict(X_test)        # integer class indices
accuracy    = model.score(X_test, H_test)  # float in [0, 1]
```


## References

- M. Aharon, M. Elad, A. Bruckstein. *"The K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation"*. IEEE Trans. Signal Processing, 54(11), 2006.
- M. Elad, R. Rubinstein, M. Zibulevsky. *"Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit"*. Technion Technical Report, 2008.
- Z. Jiang, Z. Lin, L. Davis. *"Learning A Discriminative Dictionary for Sparse Coding via Label Consistent K-SVD"*. CVPR, 2011.
- B. T. Carroll, B. M. Whitaker, W. Daley, D. V. Anderson. "Outlier Learning via Augmented Frozen Dictionaries". IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2017.
