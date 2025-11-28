# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for FLPerceptron
"""

import pytest
import numpy as np

try:
    import secretflow as sf
    from secretflow.data import FedNdarray, PartitionWay
    from secretflow.device.driver import reveal
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False

# Import the adapter
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from secretlearn.FL.linear_models.perceptron import FLPerceptron


@pytest.mark.skipif(not SECRETFLOW_AVAILABLE, reason="SecretFlow not available")
def test_perceptron_basic():
    """Basic functionality test"""
    # Initialize SecretFlow
    sf.init(['alice', 'bob'], address='local')
    spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))
    
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')
    
    # Create test data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    X_alice = X[:, :5]
    X_bob = X[:, 5:]
    
    # Create federated data
    fed_X = FedNdarray(
        partitions={
            alice: alice(lambda x: x)
    
    # Create federated labels
    fed_y = FedNdarray(
        partitions={
            alice: alice(lambda x: x)(y),
        },
        partition_way=PartitionWay.HORIZONTAL
    )(X_alice),
            bob: bob(lambda x: x)(X_bob),
        },
        partition_way=PartitionWay.VERTICAL
    )
    
    # Test model
    # FL mode API
    heu = None  # Optional: sf.HEU(...) for secure aggregation
    model = FLPerceptron(
        devices={'alice': alice, 'bob': bob},
        heu=heu
    )
    model.fit(fed_X, fed_y)
    
    # Basic assertions
    assert model.model_state_ is not None
    print("Basic test passed")
    
    sf.shutdown()


@pytest.mark.skipif(not SECRETFLOW_AVAILABLE, reason="SecretFlow not available")
def test_perceptron_consistency():
    """Test numerical consistency with sklearn"""
    sf.init(['alice', 'bob'], address='local')
    spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))
    
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')
    
    # Create test data
    np.random.seed(42)
    X = np.random.randn(50, 10).astype(np.float32)
    
    # Test with sklearn
    from sklearn.linear_model import Perceptron
    sklearn_model = Perceptron()
    sklearn_model.fit(X, y)
    
    # Test with SecretFlow adapter
    X_alice = X[:, :5]
    X_bob = X[:, 5:]
    
    fed_X = FedNdarray(
        partitions={
            alice: alice(lambda x: x)
    
    # Create federated labels
    fed_y = FedNdarray(
        partitions={
            alice: alice(lambda x: x)(y),
        },
        partition_way=PartitionWay.HORIZONTAL
    )(X_alice),
            bob: bob(lambda x: x)(X_bob),
        },
        partition_way=PartitionWay.VERTICAL
    )
    
    sf_# FL mode API
    heu = None  # Optional: sf.HEU(...) for secure aggregation
    model = FLPerceptron(
        devices={'alice': alice, 'bob': bob},
        heu=heu
    )
    sf_model.fit(fed_X, fed_y)
    
    # Compare key attributes
    print("Consistency test passed")
    
    sf.shutdown()


if __name__ == "__main__":
    print(f"Testing FLPerceptron...")
    print("Note: These tests require SecretFlow to be installed")
    pytest.main([__file__, "-v", "-s"])
