# reppi/backend.py

USE_GPU = False

if USE_GPU:
    import cupy as xp
    from cupyx.scipy import linalg
else:
    import numpy as xp
    from scipy import linalg