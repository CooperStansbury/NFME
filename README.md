# NMFE
`NMFE` is a Python class (Python 3.7.5) that is a collection of tools from other libraries designed to make working with non-negative matrix factorization a
little more friendly.


## Requirements
This module is not currently designed or tested on OSs other than OSX. `NMFE` is not written to be backwards compatible with early versions of imported libraries or Python distributions.

This library depends on the following imports:

```Python
import random
import warnings
from operator import truediv as div
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import minmax_scale as sklearn_norm
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from skimage import io
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from scipy.spatial.distance import cdist
from scipy import signal
```

More specifically, the versions I am using are:

```
sklearn version: 0.20.0
numpy version: 1.16.3
skimage version: 0.16.2
seaborn version: 0.9.0
matplotlib version: 3.1.1
scipy version: 1.3.2
```

## Use
An example of basic use is below. Call `help()` on class methods if you need more information.

```Python
from NMFE import NMFE
nmfe = NMFE(input_matrix=<your_data>,
            norm=True,
            n_components=10)

print(nmfe.compute_fit_error())
nmfe.plot_IE()
```
