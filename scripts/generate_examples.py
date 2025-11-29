#!/usr/bin/env python3
"""
Generate example files for FL/SS/SL modes
"""

import os
import re
import sys
from pathlib import Path

BASE_PATH = '/Users/xingqiangchen/secret-learn'
sys.path.insert(0, BASE_PATH)

try:
    from secretlearn.algorithm_classifier import classify_algorithm
except ImportError:
    def classify_algorithm(name):
        """Fallback classifier"""
        unsupervised = ['KMeans', 'PCA', 'DBSCAN', 'AgglomerativeClustering', 
                       'SpectralClustering', 'Birch', 'MeanShift', 'OPTICS',
                       'TruncatedSVD', 'NMF', 'FactorAnalysis', 'FastICA',
                       'LatentDirichletAllocation', 'SparsePCA', 'MiniBatchSparsePCA',
                       'IsolationForest', 'LocalOutlierFactor', 'OneClassSVM',
                       'TSNE', 'SpectralEmbedding', 'Isomap', 'MDS', 'LocallyLinearEmbedding']
        epochs = ['MLPClassifier', 'MLPRegressor', 'SGDClassifier', 'SGDRegressor',
                 'Perceptron', 'PassiveAggressiveClassifier', 'PassiveAggressiveRegressor']
        return {
            'is_unsupervised': name in unsupervised or any(u in name for u in unsupervised),
            'use_epochs': name in epochs or any(e in name for e in epochs)
        }

def camel_to_snake(name):
    """"""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def snake_to_camel(snake_str):
    """"""
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)


def get_actual_class_name(mode, category, snake_name):
    """
    Read the actual class name from the secretlearn module file.
    Returns the correct class name (e.g., SSAdaBoostRegressor not SSAdaboostRegressor)
    """
    module_path = Path(BASE_PATH) / 'secretlearn' / mode / category / f'{snake_name}.py'
    
    if not module_path.exists():
        return None
    
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find class definition: class SSXxx: or class FLXxx: or class SLXxx:
        match = re.search(rf'^class ({mode}[A-Za-z0-9]+)[\s:(]', content, re.MULTILINE)
        if match:
            return match.group(1)
    except:
        pass
    
    return None


def generate_example(algo_name, category, mode, actual_class_name=None):
    """
    Generate correct example file
    
    Parameters
    ----------
    algo_name : str
        Algorithm name, e.g., 'LinearSVC'
    category : str
        Category, e.g., 'svm'
    mode : str
        'FL', 'SS', or 'SL'
    actual_class_name : str, optional
        The actual class name from the module (to handle case differences)
    """
    snake_name = camel_to_snake(algo_name)
    
    # Use actual class name if provided, otherwise construct it
    if actual_class_name:
        class_name = actual_class_name
    else:
        class_name = f"{mode}{algo_name}"
    
    # 
    try:
        char = classify_algorithm(algo_name)
        is_unsupervised = char.get('is_unsupervised', False)
        use_epochs = char.get('use_epochs', False)
    except:
        is_unsupervised = False
        use_epochs = False
    
    # pattern
    port_base = {'FL': 9491, 'SS': 9494, 'SL': 9497}
    ports = [port_base[mode] + i for i in range(3)]
    
    # patterninitialization
    if mode == 'FL':
        init_code = """    # Create devices dict for FL mode
    devices = {"alice": alice, "bob": bob, "carol": carol}
    """
    elif mode == 'SS':
        init_code = """    # Use SPU for SS mode
    """
    else:  # SL
        init_code = """    # Create devices dict for SL mode
    devices = {"alice": alice, "bob": bob, "carol": carol}
    """
    
    # pattern
    if mode == 'FL' or mode == 'SL':
        model_init = f"model = {class_name}(devices)"
    else:  # SS
        model_init = f"model = {class_name}(spu)"
    
    #  fit 
    if is_unsupervised:
        data_code = """    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # No labels needed for unsupervised learning"""
        
        label_section = ""
        
        fit_code = f"""    {model_init}
    model.fit(fed_X)  # Unsupervised: no labels"""
        
    elif use_epochs:
        data_code = """    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 3, n_samples).astype(np.int32)  # Classification labels"""
        
        label_section = """
    # Create federated labels
    fed_y = FedNdarray(
        partitions={
            alice: alice(lambda x: x)(y),
        },
        partition_way=PartitionWay.HORIZONTAL
    )"""
        
        fit_code = f"""    {model_init}
    model.fit(fed_X, fed_y, epochs=10)  # Iterative training"""
        
    else:  # supervised non-iterative
        data_code = """    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)  # Target values"""
        
        label_section = """
    # Create federated labels
    fed_y = FedNdarray(
        partitions={
            alice: alice(lambda x: x)(y),
        },
        partition_way=PartitionWay.HORIZONTAL
    )"""
        
        fit_code = f"""    {model_init}
    model.fit(fed_X, fed_y)"""
    
    template = f'''#!/usr/bin/env python3
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Usage Example for {class_name}

This example demonstrates how to use the privacy-preserving {algo_name}
in SecretFlow's {mode} mode.
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

from secretlearn.{mode}.{category}.{snake_name} import {class_name}


def main():
    """Main example function"""
    print("="*70)
    print(f" {class_name} Usage Example")
    print("="*70)
    
    # Step 1: Initialize SecretFlow (PRODUCTION mode for SF 1.11+)
    print("\\n[1/5] Initializing SecretFlow...")
    
    # For single-node testing (simulated multi-party)
    cluster_config = {{
        'parties': {{
            'alice': {{'address': 'localhost:{ports[0]}', 'listen_addr': '0.0.0.0:{ports[0]}'}},
            'bob': {{'address': 'localhost:{ports[1]}', 'listen_addr': '0.0.0.0:{ports[1]}'}},
            'carol': {{'address': 'localhost:{ports[2]}', 'listen_addr': '0.0.0.0:{ports[2]}'}},
        }},
        'self_party': 'alice'
    }}
    
    # Initialize with PRODUCTION mode (SF 1.11+ removes Ray/SIMULATION mode)
    sfd.init(DISTRIBUTION_MODE.PRODUCTION, cluster_config=cluster_config)
    
    # Create SPU device
    spu_config = sf.utils.testing.cluster_def(
        parties=['alice', 'bob', 'carol'],
        runtime_config={{'protocol': 'ABY3', 'field': 'FM64'}}
    )
    spu = sf.SPU(spu_config)
    
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')
    carol = sf.PYU('carol')
    print("  ✓ SecretFlow initialized (PRODUCTION mode)")
    
    # Step 2: Create sample data
    print("\\n[2/5] Creating sample data...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
{data_code}
    
    # Partition data vertically
    X_alice = X[:, 0:5]
    X_bob = X[:, 5:10]
    X_carol = X[:, 10:15]
    
    print(f"  ✓ Data shape: {{n_samples}} samples × {{n_features}} features")
    print(f"  ✓ Alice: {{X_alice.shape}}, Bob: {{X_bob.shape}}, Carol: {{X_carol.shape}}")
    
    # Step 3: Create federated data
    print("\\n[3/5] Creating federated data...")
    fed_X = FedNdarray(
        partitions={{
            alice: alice(lambda x: x)(X_alice),
            bob: bob(lambda x: x)(X_bob),
            carol: carol(lambda x: x)(X_carol),
        }},
        partition_way=PartitionWay.VERTICAL
    ){label_section}
    print("  ✓ Federated data created")
    
    # Step 4: Train model
    print("\\n[4/5] Training {class_name}...")
    print("  Note: All computation with privacy protection")
    
    import time
    start_time = time.time()
    
{init_code}
{fit_code}
    
    training_time = time.time() - start_time
    print(f"  ✓ Training completed in {{training_time*1000:.2f}}ms")
    
    # Step 5: Make predictions (if applicable)
    print("\\n[5/5] Model trained successfully!")
    print("  ✓ Privacy: Fully protected")
    print(f"  ✓ Performance: {{training_time*1000:.2f}}ms")
    
    # Cleanup
    sf.shutdown()
    print("\\nExample completed!")


if __name__ == "__main__":
    main()
'''
    
    return template

def get_category_from_import(content):
    """ import Extract"""
    match = re.search(r'from secretlearn\.[A-Z]+\.(\w+)\.', content)
    if match:
        return match.group(1)
    return None

def regenerate_problematic_examples(base_path):
    """ example file"""
    print("="*90)
    print(" Example file")
    print("="*90)
    print()
    
    regenerated = 0
    
    for mode in ['FL', 'SS', 'SL']:
        examples_path = Path(base_path) / 'examples' / mode
        
        if not examples_path.exists():
            continue
        
        print(f"\n【{mode} pattern】")
        mode_count = 0
        
        for py_file in sorted(examples_path.glob('*.py')):
            if py_file.name == '__init__.py':
                continue
            
            try:
                # file
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 
                has_issue = False
                
                # 1. 
                if '{algo_name}' in content:
                    has_issue = True
                
                # 2. error（）
                try:
                    compile(content, py_file.name, 'exec')
                except SyntaxError:
                    has_issue = True
                
                # 3. class nameerror
                # ：from secretlearn.SL.svm.linear_svc import FLLinearSVC
                # ：from secretlearn.SL.svm.linear_svc import SLLinearSVC
                import_match = re.search(
                    rf'from secretlearn\.{mode}\.(\w+)\.(\w+) import ([A-Z]\w+)',
                    content
                )
                if import_match:
                    category, module, class_name = import_match.groups()
                    expected_class = f"{mode}{snake_to_camel(module)}"
                    if class_name != expected_class:
                        has_issue = True
                
                if not has_issue:
                    continue
                
                # Extractinfo
                filename = py_file.stem
                algo_name = snake_to_camel(filename)
                category = get_category_from_import(content)
                
                if not category:
                    print(f"  ⚠️  : {py_file.name}")
                    continue
                
                # 
                new_content = generate_example(algo_name, category, mode)
                
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                mode_count += 1
                regenerated += 1
                
                if mode_count <= 10:
                    print(f"  {py_file.name}")
                
            except Exception as e:
                print(f"   {py_file.name}: {str(e)}")
        
        if mode_count > 10:
            print(f"  ...  {mode_count - 10} file")
        
        if mode_count > 0:
            print(f"  {mode} : {mode_count} file")
    
    print(f"\n: {regenerated} file")
    return regenerated

def regenerate_ss_only(base_path, force=False):
    """Regenerate only SS example files"""
    print("=" * 70)
    print("Regenerating SS example files")
    print("=" * 70)
    
    examples_path = Path(base_path) / 'examples' / 'SS'
    
    if not examples_path.exists():
        print(f"Error: {examples_path} does not exist")
        return 0
    
    regenerated = 0
    errors = []
    
    for py_file in sorted(examples_path.glob('*.py')):
        if py_file.name == '__init__.py':
            continue
        
        try:
            # Read existing file to get category
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if needs regeneration
            needs_regen = force
            if not force:
                try:
                    compile(content, py_file.name, 'exec')
                except SyntaxError:
                    needs_regen = True
            
            if not needs_regen:
                continue
            
            # Extract info
            filename = py_file.stem
            algo_name = snake_to_camel(filename)
            category = get_category_from_import(content)
            
            if not category:
                # Try to guess category from secretlearn module structure
                category = guess_category(algo_name)
                if not category:
                    errors.append((py_file.name, "Cannot determine category"))
                    continue
            
            # Get the actual class name from the module file
            actual_class_name = get_actual_class_name('SS', category, filename)
            
            # Generate new content with the correct class name
            new_content = generate_example(algo_name, category, 'SS', actual_class_name)
            
            # Verify syntax before writing
            try:
                compile(new_content, py_file.name, 'exec')
            except SyntaxError as e:
                errors.append((py_file.name, f"Generated code has syntax error: {e}"))
                continue
            
            with open(py_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            regenerated += 1
            print(f"✓ {py_file.name}")
            
        except Exception as e:
            errors.append((py_file.name, str(e)))
    
    print("=" * 70)
    print(f"Regenerated: {regenerated} files")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for fname, err in errors[:10]:
            print(f"  ✗ {fname}: {err[:60]}")
    
    return regenerated


def guess_category(algo_name):
    """Guess the category based on algorithm name"""
    name_lower = algo_name.lower()
    
    category_map = {
        'linear_models': ['linear', 'ridge', 'lasso', 'elastic', 'sgd', 'perceptron', 
                         'logistic', 'bayesian', 'ard', 'lars', 'omp', 'huber',
                         'ransac', 'theilsen', 'passiveaggressive', 'tweedie', 
                         'poisson', 'gamma', 'quantile'],
        'ensemble': ['forest', 'boosting', 'adaboost', 'bagging', 'voting', 'stacking',
                    'histgradient', 'extratree', 'isolation'],
        'svm': ['svc', 'svr', 'nusvc', 'nusvr', 'linearsvc', 'linearsvr', 'onesvm'],
        'tree': ['decision', 'extratree'],
        'neighbors': ['kneighbors', 'radiusneighbors', 'nearestneighbors', 
                     'nearestcentroid', 'localoutlier', 'knn'],
        'clustering': ['kmeans', 'dbscan', 'agglomerative', 'spectralclustering',
                      'birch', 'meanshift', 'optics', 'affinity', 'featureagglom',
                      'minibatchkmeans', 'bisectingkmeans'],
        'decomposition': ['pca', 'svd', 'nmf', 'ica', 'lda', 'factor', 'sparse',
                         'dictionary', 'minibatchdictionary', 'minibatchnmf',
                         'truncatedsvd', 'kernelpca', 'incrementalpca'],
        'preprocessing': ['scaler', 'normalizer', 'binarizer', 'encoder', 
                         'transformer', 'imputer', 'discretizer', 'polynomial',
                         'spline', 'kbins', 'quantile', 'power', 'maxabs',
                         'minmax', 'standard', 'robust', 'label', 'ordinal',
                         'onehot', 'target', 'function'],
        'naive_bayes': ['naive', 'bayes', 'gaussian', 'multinomial', 'bernoulli',
                       'complement', 'categorical'],
        'neural_network': ['mlp', 'perceptron', 'bernoulli'],
        'discriminant_analysis': ['lda', 'qda', 'lineardiscriminant', 'quadraticdiscriminant'],
        'gaussian_process': ['gaussian', 'gp'],
        'manifold': ['tsne', 'isomap', 'mds', 'spectralembedding', 'locallylinear'],
        'feature_selection': ['selectk', 'selectpercentile', 'selectfpr', 'selectfdr',
                             'selectfwe', 'rfe', 'rfecv', 'variancethreshold',
                             'selectfrommodel', 'sequentialfeature', 'genericunivariateselect'],
        'covariance': ['covariance', 'empirical', 'shrunk', 'oas', 'ledoitwolf',
                      'mincovdet', 'graphicallasso', 'ellipticenvelope'],
        'kernel_approximation': ['rbfsampler', 'nystroem', 'additivech', 'skewedchi',
                                'polynomial'],
        'multiclass': ['onevsrest', 'onevsone', 'outputcode'],
        'multioutput': ['multioutput', 'classifierchain', 'regressorchain'],
        'calibration': ['calibrated'],
        'isotonic': ['isotonic'],
        'cross_decomposition': ['pls', 'cca'],
        'impute': ['simpleimputer', 'knnimputer', 'iterativeimputer'],
        'semi_supervised': ['labelpropagation', 'labelspreading', 'selftraining'],
        'kernel_ridge': ['kernelridge'],
        'random_projection': ['gaussianrandomprojection', 'sparserandomprojection'],
        'dummy': ['dummy'],
        'feature_extraction': ['dictvectorizer', 'featurehasher'],
        'mixture': ['gaussianmixture', 'bayesiangaussianmixture'],
        'anomaly_detection': ['isolationforest', 'localoutlierfactor', 'oneclasssvm'],
    }
    
    for category, keywords in category_map.items():
        for keyword in keywords:
            if keyword in name_lower:
                return category
    
    return 'clustering'  # Default fallback


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate example files')
    parser.add_argument('--mode', choices=['FL', 'SS', 'SL', 'all'], default='SS',
                       help='Which mode to regenerate (default: SS)')
    parser.add_argument('--force', action='store_true',
                       help='Force regenerate all files, not just those with errors')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Example File Generator")
    print("=" * 70)
    print()
    
    if args.mode == 'SS':
        count = regenerate_ss_only(BASE_PATH, force=args.force)
    elif args.mode == 'all':
        count = regenerate_problematic_examples(BASE_PATH)
    else:
        print(f"Mode {args.mode} not yet implemented, use SS or all")
        return
    
    print()
    print("=" * 70)
    print(f"Completed! Regenerated {count} files")
    print("=" * 70)


if __name__ == '__main__':
    main()

