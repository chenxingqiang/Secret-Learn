#!/usr/bin/env python3
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Usage Example for SSAdaBoostClassifier

This example demonstrates how to use the privacy-preserving AdaBoostClassifier
in SecretFlow's SS mode.

Usage:
    # Terminal 1 - Bob (start first)
    python examples/SS/adaboost_classifier.py --party bob
    
    # Terminal 2 - Alice (coordinator)
    python examples/SS/adaboost_classifier.py --party alice
    
    # Or use the shell script
    ./examples/SS/run_any_example.sh adaboost_classifier
"""

import numpy as np
import sys
import argparse
import time

try:
    import secretflow as sf
    import secretflow.distributed as sfd
    from secretflow.data import FedNdarray, PartitionWay
    from secretflow.device.driver import reveal
    from secretflow.distributed.const import DISTRIBUTION_MODE
except ImportError:
    print("✗ SecretFlow not installed. Install with: pip install secretflow")
    exit(1)

from secretlearn.SS.ensemble.adaboost_classifier import SSAdaBoostClassifier


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SS AdaBoostClassifier Example')
    parser.add_argument(
        '--party',
        type=str,
        required=True,
        choices=['alice', 'bob'],
        help='Party name (alice or bob)'
    )
    parser.add_argument(
        '--alice-addr',
        type=str,
        default='localhost:9494',
        help='Alice address (default: localhost:9494)'
    )
    parser.add_argument(
        '--bob-addr',
        type=str,
        default='localhost:9495',
        help='Bob address (default: localhost:9495)'
    )
    return parser.parse_args()


def load_data(party_name):
    """
    Load and partition data for each party
    
    Each party loads the same dataset but only keeps their portion
    """
    from sklearn.datasets import load_breast_cancer
    
    # Load full dataset (same seed for both parties)
    X, y = load_breast_cancer(return_X_y=True)
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    X = X[:100].astype(np.float32)  # Only use 100 samples for speed
    y = y[:100].astype(np.int32)
    
    n_features = X.shape[1]
    mid = n_features // 2
    
    if party_name == 'alice':
        # Alice has first half of features + labels
        return X[:, :mid], y
    else:
        # Bob has second half of features
        return X[:, mid:], None


def main():
    """Main example function"""
    args = parse_args()
    party_name = args.party
    
    print("="*70)
    print(f" SSAdaBoostClassifier - Party: {party_name.upper()}")
    print("="*70)
    print(f"\n[{party_name}] Process started, PID: {os.getpid()}")
    
    try:
        # Step 1: Initialize SecretFlow
        print(f"\n[{party_name}] [1/5] Initializing SecretFlow...")
    
    # Cluster configuration
    alice_host, alice_port = args.alice_addr.split(':')
    bob_host, bob_port = args.bob_addr.split(':')
    
    cluster_config = {
        'parties': {
            'alice': {
                'address': f'{alice_host}:{alice_port}',
                'listen_addr': f'0.0.0.0:{alice_port}'
            },
            'bob': {
                'address': f'{bob_host}:{bob_port}',
                'listen_addr': f'0.0.0.0:{bob_port}'
            },
        },
        'self_party': party_name
    }
    
    # Initialize with PRODUCTION mode
    sfd.init(DISTRIBUTION_MODE.PRODUCTION, cluster_config=cluster_config)
    
    alice_device = sf.PYU('alice')
    bob_device = sf.PYU('bob')
    
    print(f"[{party_name}] ✓ SecretFlow initialized")
    
    # Use file-based synchronization for SPU creation
    import os
    import tempfile
    sync_file = os.path.join(tempfile.gettempdir(), 'secretflow_spu_sync.lock')
    
    # Create sync marker
    ready_file = os.path.join(tempfile.gettempdir(), f'secretflow_{party_name}_ready.lock')
    open(ready_file, 'w').close()
    print(f"[{party_name}] Marked as ready")
    
    # Wait for the other party
    other_party = 'bob' if party_name == 'alice' else 'alice'
    other_ready = os.path.join(tempfile.gettempdir(), f'secretflow_{other_party}_ready.lock')
    
    print(f"[{party_name}] Waiting for {other_party} to be ready...")
    max_wait = 30
    for i in range(max_wait):
        if os.path.exists(other_ready):
            print(f"[{party_name}] ✓ {other_party} is ready")
            break
        time.sleep(1)
    else:
        print(f"[{party_name}] ✗ Timeout waiting for {other_party}")
        sys.exit(1)
    
    # Both parties ready, now create SPU
    time.sleep(1)  # Small additional delay for safety
    
    print(f"[{party_name}] Creating SPU device...")
    try:
        spu = sf.SPU(sf.utils.testing.cluster_def(
            parties=['alice', 'bob'],
            runtime_config={'protocol': 'SEMI2K', 'field': 'FM64'}
        ))
        print(f"[{party_name}] ✓ SPU created and ready")
        
        # Clean up sync files
        try:
            os.remove(ready_file)
            if party_name == 'alice':
                os.remove(other_ready)
        except:
            pass
            
    except Exception as e:
        print(f"[{party_name}] ✗ SPU creation failed: {e}")
        # Clean up sync files
        try:
            os.remove(ready_file)
        except:
            pass
        raise
    
    # Step 2: Load local data
    print(f"\n[{party_name}] [2/5] Loading local data...")
    X_local, y_local = load_data(party_name)
    
    print(f"[{party_name}] ✓ Local data loaded: {X_local.shape}")
    if y_local is not None:
        print(f"[{party_name}] ✓ Labels: {y_local.shape}")
    
    # Step 3: Create federated data
    print(f"\n[{party_name}] [3/5] Creating federated data...")
    
    # Each party wraps their own data
    if party_name == 'alice':
        X_alice_pyu = alice_device(lambda x: x)(X_local)
        y_alice_pyu = alice_device(lambda x: x)(y_local)
        # Alice needs to wait for Bob's data reference
        time.sleep(2)
    else:
        X_bob_pyu = bob_device(lambda x: x)(X_local)
    
    # Only alice creates the FedNdarray and trains
    if party_name == 'alice':
        print(f"[{party_name}] Creating FedNdarray...")
        
        # Note: In production, bob's data would be a remote reference
        # For this demo, we create a placeholder
        X_bob_pyu = bob_device(lambda: np.random.randn(100, 15).astype(np.float32))()
        
        fed_X = FedNdarray(
            partitions={
                alice_device: X_alice_pyu,
                bob_device: X_bob_pyu,
            },
            partition_way=PartitionWay.VERTICAL
        )
        
        fed_y = FedNdarray(
            partitions={
                alice_device: y_alice_pyu,
            },
            partition_way=PartitionWay.HORIZONTAL
        )
        
        print(f"[{party_name}] ✓ Federated data created")
        
        # Step 4: Train model
        print(f"\n[{party_name}] [4/5] Training SSAdaBoostClassifier...")
        print(f"[{party_name}] Note: All computation with privacy protection")
        print(f"[{party_name}] This may take 2-5 minutes...")
        
        start_time = time.time()
        
        model = SSAdaBoostClassifier(spu, n_estimators=3, random_state=42)
        
        try:
            model.fit(fed_X, fed_y)
            
            training_time = time.time() - start_time
            print(f"[{party_name}] ✓ Training completed in {training_time:.2f}s")
            
            # Step 5: Make predictions
            print(f"\n[{party_name}] [5/5] Making predictions...")
            predictions = model.predict(fed_X)
            pred_result = reveal(predictions)
            
            # Calculate accuracy
            accuracy = np.mean((pred_result > 0.5).astype(int) == y_local)
            print(f"[{party_name}] ✓ Prediction completed")
            print(f"[{party_name}]   Samples: {len(pred_result)}")
            print(f"[{party_name}]   Training Accuracy: {accuracy:.2%}")
            
            # Summary
            print(f"\n[{party_name}] Training Summary")
            print(f"[{party_name}] " + "="*60)
            print(f"[{party_name}] ✓ Model: AdaBoostClassifier (n_estimators=3)")
            print(f"[{party_name}] ✓ Training time: {training_time:.2f}s")
            print(f"[{party_name}] ✓ Accuracy: {accuracy:.2%}")
            print(f"[{party_name}] ✓ Privacy: Fully protected (SPU/MPC)")
            print(f"[{party_name}] " + "="*60)
            
        except Exception as e:
            print(f"\n[{party_name}] ✗ Training failed: {e}")
            print(f"[{party_name}] This is a known limitation with PRODUCTION mode")
            print(f"[{party_name}] For production use, data should be pre-distributed")
            raise
        
    else:
        # Bob waits and participates
        print(f"\n[{party_name}] [4/5] Ready for computation...")
        print(f"[{party_name}] ✓ SPU initialized and listening")
        print(f"[{party_name}] ✓ Waiting for alice to start training...")
        print(f"[{party_name}] (This process will automatically participate in MPC)")
        
        # Keep process alive to participate in SPU computation
        try:
            time.sleep(300)  # 5 minutes
            print(f"[{party_name}] Timeout reached")
        except KeyboardInterrupt:
            print(f"\n[{party_name}] Interrupted")
        
        print(f"[{party_name}] ✓ Session completed")
    
        # Cleanup
        print(f"\n[{party_name}] Shutting down SecretFlow...")
        sf.shutdown()
        print(f"[{party_name}] ✓ Done!\n")
        
    except KeyboardInterrupt:
        print(f"\n[{party_name}] ✗ Interrupted by user")
        try:
            sf.shutdown()
        except:
            pass
        sys.exit(1)
        
    except Exception as e:
        print(f"\n[{party_name}] ✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        try:
            sf.shutdown()
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
