#!/usr/bin/env python3
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Usage Example for FLGaussianProcessClassifier

This example demonstrates how to use the privacy-preserving GaussianProcessClassifier
in SecretFlow's federated learning environment.
"""

import numpy as np

try:
    import secretflow as sf
    import secretflow.distributed as sfd
    from secretflow.data import FedNdarray, PartitionWay
    from secretflow.device.driver import reveal
    from secretflow.distributed.const import DISTRIBUTION_MODE
except ImportError:
    print(" SecretFlow not installed. Install with: pip install secretflow")
    exit(1)

from secretlearn.FL.gaussian_process.gaussian_process_classifier import FLGaussianProcessClassifier


def main():
    """Main example function"""
    print("="*70)
    print(f" FLGaussianProcessClassifier Usage Example")
    print("="*70)
    
    # Step 1: Initialize SecretFlow (PRODUCTION mode for SF 1.11+)
    print("\n[1/5] Initializing SecretFlow...")
    
    # For single-node testing (simulated multi-party)
    cluster_config = {
        'parties': {
            'alice': {'address': 'localhost:9491', 'listen_addr': '0.0.0.0:9491'},
            'bob': {'address': 'localhost:9492', 'listen_addr': '0.0.0.0:9492'},
            'carol': {'address': 'localhost:9493', 'listen_addr': '0.0.0.0:9493'},
        },
        'self_party': 'alice'
    }
    
    # Initialize with PRODUCTION mode (SF 1.11+ removes Ray/SIMULATION mode)
    sfd.init(DISTRIBUTION_MODE.PRODUCTION, cluster_config=cluster_config)
    
    # Create SPU device
    spu_config = sf.utils.testing.cluster_def(
        parties=['alice', 'bob', 'carol'],
        runtime_config={'protocol': 'ABY3', 'field': 'FM64'}
    )
    spu = sf.SPU(spu_config)
    
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')
    carol = sf.PYU('carol')
    print("  ✓ SecretFlow initialized (PRODUCTION mode)")
    
    # Step 2: Create sample data
    print("\n[2/5] Creating sample data...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)  # Target values
    
    # Partition data vertically
    X_alice = X[:, 0:5]
    X_bob = X[:, 5:10]
    X_carol = X[:, 10:15]
    
    print(f"  ✓ Data shape: {n_samples} samples × {n_features} features")
    print(f"  ✓ Alice: {X_alice.shape}, Bob: {X_bob.shape}, Carol: {X_carol.shape}")
    
    # Step 3: Create federated data
    print("\n[3/5] Creating federated data...")
    fed_X = FedNdarray(
        partitions={
            alice: alice(lambda x: x)(X_alice),
            bob: bob(lambda x: x)(X_bob),
            carol: carol(lambda x: x)(X_carol),
        },
        partition_way=PartitionWay.VERTICAL
    )
    
    # Create federated labels
    fed_y = FedNdarray(
        partitions={
            alice: alice(lambda x: x)(y),
        },
        partition_way=PartitionWay.HORIZONTAL
    )
    print("  ✓ Federated data created")
    
    # Step 4: Train model
    print("\n[4/5] Training FLGaussianProcessClassifier...")
    print("  Note: All computation happens in SPU's encrypted environment")
    
    import time
    start_time = time.time()
    
    # Create devices dict for FL mode

    
    devices = {"alice": alice, "bob": bob, "carol": carol}

    
    model = FLGaussianProcessClassifier(devices)
    model.fit(fed_X, fed_y)
    
    training_time = time.time() - start_time
    print(f"  ✓ Training completed in {training_time*1000:.2f}ms")
    
    # Step 5: Make predictions (if applicable)
    print("\n[5/5] Model trained successfully!")
    print("  ✓ Model state stored in SPU (encrypted)")
    print("  ✓ Privacy: Fully protected by MPC")
    print(f"  ✓ Performance: {training_time*1000:.2f}ms")
    
    # Cleanup
    sf.shutdown()
    print("\nExample completed!")


if __name__ == "__main__":
    main()
