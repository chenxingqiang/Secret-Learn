#!/usr/bin/env python3
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Usage Example for SSQuadraticdiscriminantanalysis

Usage:
    Terminal 1: python quadratic_discriminant_analysis.py --party bob
    Terminal 2: python quadratic_discriminant_analysis.py --party alice

This example demonstrates how to use the privacy-preserving QuadraticDiscriminantAnalysis
in SecretFlow's federated learning environment.
"""

import numpy as np
import sys
import argparse
import time
import os

try:
    import secretflow as sf
    import secretflow.distributed as sfd
    from secretflow.data import FedNdarray, PartitionWay
    from secretflow.device.driver import reveal
    from secretflow.distributed.const import DISTRIBUTION_MODE
except ImportError:
    print(" SecretFlow not installed. Install with: pip install secretflow")
    exit(1)

from secretlearn.SS.discriminant_analysis.quadratic_discriminant_analysis import SSQuadraticdiscriminantanalysis


def parse_args():
    parser = argparse.ArgumentParser(description='SS SSQuadraticdiscriminantanalysis')
    parser.add_argument('--party', required=True, choices=['alice', 'bob'])
    parser.add_argument('--alice-addr', default='localhost:9494')
    parser.add_argument('--bob-addr', default='localhost:9495')
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    party_name = args.party
    
    print("="*70)
    print(f" SSQuadraticdiscriminantanalysis - Party: {party_name.upper()}")
    print("="*70)
    print(f"
[{party_name}] PID: {os.getpid()}")
    
    try:
        print(f"
[{party_name}] [1/5] Initializing...")
        
        # Cluster config
        alice_host, alice_port = args.alice_addr.split(':')
        bob_host, bob_port = args.bob_addr.split(':')
        
        cluster_config = {
            'parties': {
                'alice': {'address': f'{alice_host}:{alice_port}', 'listen_addr': f'0.0.0.0:{alice_port}'},
                'bob': {'address': f'{bob_host}:{bob_port}', 'listen_addr': f'0.0.0.0:{bob_port}'},
            },
            'self_party': party_name
        }
        
        sfd.init(DISTRIBUTION_MODE.PRODUCTION, cluster_config=cluster_config)
        
        alice = sf.PYU('alice')
        bob = sf.PYU('bob')
        
        # Sync mechanism
        ready_file = f'/tmp/sf_{party_name}.lock'
        other_ready = f'/tmp/sf_{"bob" if party_name == "alice" else "alice"}.lock'
        open(ready_file, 'w').close()
        
        print(f"[{party_name}] Waiting for peer...")
        for _ in range(30):
            if os.path.exists(other_ready):
                break
            time.sleep(1)
        else:
            print(f"[{party_name}] ✗ Timeout")
            sys.exit(1)
        
        time.sleep(1)
        
        # SPU
        spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob'], runtime_config={'protocol': 'SEMI2K'}))
        print(f"[{party_name}] ✓ Initialized")
        try:
            os.remove(ready_file)
        except:
            pass
        
        if party_name == 'alice':
            # Alice runs the training
            print(f"
[{party_name}] [2/5] Preparing data...")    
    # Step 2: Create sample data
    print("\n[2/5] Creating sample data...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
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
            bob: bob(lambda x: x)(X_bob): carol(lambda x: x)(X_carol),
        },
        partition_way=PartitionWay.VERTICAL
    )
    print("  ✓ Federated data created")
    
    # Step 4: Train model
    print("\n[4/5] Training SSQuadraticdiscriminantanalysis...")
    print("  Note: All computation happens in SPU's encrypted environment")
    
    import time
    start_time = time.time()
    
    model = SSQuadraticdiscriminantanalysis(spu)
    model.fit(fed_X)
    
    training_time = time.time() - start_time
    print(f"  ✓ Training completed in {training_time*1000:.2f}ms")
    
    # Step 5: Make predictions (if applicable)
    print("\n[5/5] Model trained successfully!")
    print("  ✓ Model state stored in SPU (encrypted)")
    print("  ✓ Privacy: Fully protected by MPC")
    print(f"  ✓ Performance: {training_time*1000:.2f}ms")
    else:
        # Bob waits and participates
        print(f"
[{{party_name}}] Waiting for alice...")
        time.sleep(300)
        print(f"[{{party_name}}] ✓ Done")

    
    # Cleanup
    sf.shutdown()
    print("\nExample completed!")


if __name__ == "__main__":
    main()
