# SecretFlow Examples

⚠️  **Note:** The example files in this directory were auto-generated from old templates and may contain errors.

**We recommend deleting these examples** and referring to the implementation files directly for correct usage.

## Correct Usage

### Importing from implementation files

```python
# FL Mode
from secretlearn.FL.clustering.kmeans import FLKMeans
from secretlearn.FL.linear_models.linearregression import FLLinearRegression
from secretlearn.FL.decomposition.pca import FLPCA

# SL Mode
from secretlearn.SL.clustering.kmeans import SLKMeans

# SS Mode
from secretlearn.SS.clustering.kmeans import SSKMeans
```

### FL Mode Example

```python
import secretflow as sf
import numpy as np
from secretflow.data import FedNdarray, PartitionWay

# Initialize
sf.init(['alice', 'bob'])
alice = sf.PYU('alice')
bob = sf.PYU('bob')
heu = None  # Optional: sf.HEU(...) for secure aggregation

# Create data
X = np.random.randn(100, 10)
X_alice = X[:, :5]
X_bob = X[:, 5:]

fed_X = FedNdarray(
    partitions={
        alice: alice(lambda x: x)(X_alice),
        bob: bob(lambda x: x)(X_bob),
    },
    partition_way=PartitionWay.VERTICAL
)

# Unsupervised learning (e.g. KMeans, PCA)
from secretlearn.FL.clustering.kmeans import FLKMeans

model = FLKMeans(
    devices={'alice': alice, 'bob': bob},
    heu=heu,
    n_clusters=3
)
model.fit(fed_X)  # No y required

# Supervised learning (e.g. LinearRegression)
from secretlearn.FL.linear_models.linearregression import FLLinearRegression

y = np.random.randn(100)
fed_y = FedNdarray(
    partitions={alice: alice(lambda x: x)(y)},
    partition_way=PartitionWay.HORIZONTAL
)

model = FLLinearRegression(
    devices={'alice': alice, 'bob': bob},
    heu=heu
)
model.fit(fed_X, fed_y)  # Supervised learning requires y
predictions = model.predict(fed_X)
```

### SS Mode Example

```python
spu = sf.SPU(...)

# SS mode using spu
from secretlearn.SS.clustering.kmeans import SSKMeans

model = SSKMeans(spu=spu, n_clusters=3)
model.fit(fed_X)
```

### Suggestions

**We recommend deleting the example files and referring to the implementation files directly for correct usage.**

Each implementation file has complete documentation and examples!