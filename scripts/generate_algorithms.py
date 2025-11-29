#!/usr/bin/env python3
"""
Smart Batch Algorithm Generator

Uses algorithm classifier and template generator to automatically
select correct templates based on algorithm type:
- Unsupervised learning (clustering, dimensionality reduction, preprocessing)
- Supervised non-iterative
- Supervised iterative (partial_fit)

Features:
- Auto-detects algorithm characteristics
- Generates correct fit() signatures
- Creates proper methods for each type
- Supports FL/SS/SL modes with correct patterns
- SS mode uses SPU-based computation (not devices iteration)
"""

import os
import sys
import re

# Add secretlearn 
sys.path.insert(0, '/Users/xingqiangchen/jax-sklearn')

from secretlearn.algorithm_classifier import classify_algorithm
from secretlearn.template_generator import generate_template

# 
MISSING_ALGORITHMS = {
    'linear_models': {
        'display_name': 'Linear Models',
        'algorithms': [
            'ARDRegression', 'BayesianRidge', 'GammaRegressor', 'Lars', 'LarsCV',
            'LassoLars', 'LassoLarsCV', 'LassoLarsIC', 'LogisticRegressionCV',
            'MultiTaskElasticNet', 'MultiTaskElasticNetCV', 'MultiTaskLasso',
            'MultiTaskLassoCV', 'OrthogonalMatchingPursuit', 'OrthogonalMatchingPursuitCV',
            'PoissonRegressor', 'QuantileRegressor', 'RidgeClassifierCV',
            'SGDOneClassSVM', 'TheilSenRegressor', 'TweedieRegressor'
        ]
    },
    'svm': {
        'display_name': 'SVM',
        'algorithms': ['OneClassSVM']
    },
    'tree': {
        'display_name': 'Tree',
        'algorithms': ['ExtraTreeClassifier', 'ExtraTreeRegressor']
    },
    'ensemble': {
        'display_name': 'Ensemble',
        'algorithms': ['IsolationForest', 'RandomTreesEmbedding', 'StackingClassifier', 'StackingRegressor']
    },
    'naive_bayes': {
        'display_name': 'Naive Bayes',
        'algorithms': ['LabelBinarizer']
    },
    'neighbors': {
        'display_name': 'Neighbors',
        'algorithms': ['KNeighborsTransformer', 'KernelDensity', 'NearestNeighbors',
                      'NeighborhoodComponentsAnalysis', 'RadiusNeighborsTransformer']
    },
    'neural_network': {
        'display_name': 'Neural Network',
        'algorithms': ['BernoulliRBM']
    },
    'discriminant_analysis': {
        'display_name': 'Discriminant Analysis',
        'algorithms': ['StandardScaler']  # ， preprocessing
    },
    'semi_supervised': {
        'display_name': 'Semi Supervised',
        'algorithms': ['SelfTrainingClassifier']
    },
    'clustering': {
        'display_name': 'Clustering',
        'algorithms': ['BisectingKMeans', 'FeatureAgglomeration', 'HDBSCAN',
                      'OPTICS', 'SpectralBiclustering', 'SpectralCoclustering']
    },
    'mixture': {
        'display_name': 'Mixture',
        'algorithms': ['BayesianGaussianMixture', 'GaussianMixture']
    },
    'decomposition': {
        'display_name': 'Decomposition',
        'algorithms': ['DictionaryLearning', 'LatentDirichletAllocation',
                      'MiniBatchSparsePCA', 'SparseCoder', 'SparsePCA']
    },
    'random_projection': {
        'display_name': 'Random Projection',
        'algorithms': ['BaseRandomProjection', 'GaussianRandomProjection', 'SparseRandomProjection']
    },
    'kernel_approximation': {
        'display_name': 'Kernel Approximation',
        'algorithms': ['AdditiveChi2Sampler', 'Nystroem', 'PolynomialCountSketch',
                      'RBFSampler', 'SkewedChi2Sampler']
    },
    'preprocessing': {
        'display_name': 'Preprocessing',
        'algorithms': ['FunctionTransformer', 'KernelCenterer', 'LabelBinarizer',
                      'MultiLabelBinarizer', 'OneHotEncoder', 'TargetEncoder']
    },
    'impute': {
        'display_name': 'Impute',
        'algorithms': ['KNNImputer', 'MissingIndicator', 'SimpleImputer']
    },
    'feature_extraction': {
        'display_name': 'Feature Extraction',
        'algorithms': ['DictVectorizer', 'FeatureHasher']
    },
    'feature_selection': {
        'display_name': 'Feature Selection',
        'algorithms': ['GenericUnivariateSelect', 'RFECV', 'SelectFdr',
                      'SelectFpr', 'SelectFwe', 'SelectPercentile']
    },
    'covariance': {
        'display_name': 'Covariance',
        'algorithms': ['GraphicalLasso', 'GraphicalLassoCV', 'OAS']
    },
}

# sklearn module
MODULE_MAPPING = {
    'linear_models': 'linear_model',
    'clustering': 'cluster',
    'naive_bayes': 'naive_bayes',
    'neural_network': 'neural_network',
}

def camel_to_snake(name):
    """"""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def get_sklearn_module_name(category):
    """ sklearn module"""
    return MODULE_MAPPING.get(category, category)

def generate_algorithm_smart(algo_name, category, mode, base_path):
    """
     - 

    Parameters
    ----------
    algo_name : str
        ， 'KMeans'
    category : str
        ， 'clustering'
    mode : str
        pattern，'FL', 'SS',  'SL'
    base_path : str
        
    """
    # 
    mode_dir = os.path.join(base_path, f'secretlearn/{mode}/{category}')
    os.makedirs(mode_dir, exist_ok=True)

    # file
    filename = camel_to_snake(algo_name) + '.py'
    filepath = os.path.join(mode_dir, filename)

    # file
    if os.path.exists(filepath):
        return False, ""

    #  sklearn module
    sklearn_module = get_sklearn_module_name(category)

    # 
    try:
        characteristics = classify_algorithm(algo_name)
        template_type = characteristics.get('recommended_implementation', 'supervised_non_iterative')

        # 
        if mode.upper() == 'FL':
            code = generate_template(algo_name, sklearn_module, characteristics, 'fl')
        elif mode.upper() == 'SS':
            code = generate_ss_template_smart(algo_name, sklearn_module, characteristics)
        elif mode.upper() == 'SL':
            code = generate_sl_template_smart(algo_name, sklearn_module, characteristics)
        else:
            return False, f"pattern: {mode}"

        # file
        with open(filepath, 'w') as f:
            f.write(code)

        return True, template_type

    except Exception as e:
        return False, f"error: {str(e)}"

def generate_ss_template_smart(algo_name, module_name, characteristics):
    """ SS pattern"""
    is_unsupervised = characteristics.get('is_unsupervised', False)
    use_epochs = characteristics.get('use_epochs', False)

    if is_unsupervised:
        return generate_ss_unsupervised_template(algo_name, module_name)
    elif use_epochs:
        return generate_ss_iterative_template(algo_name, module_name)
    else:
        return generate_ss_non_iterative_template(algo_name, module_name)

def generate_sl_template_smart(algo_name, module_name, characteristics):
    """ SL pattern"""
    is_unsupervised = characteristics.get('is_unsupervised', False)
    use_epochs = characteristics.get('use_epochs', False)

    if is_unsupervised:
        return generate_sl_unsupervised_template(algo_name, module_name)
    elif use_epochs:
        return generate_sl_iterative_template(algo_name, module_name)
    else:
        return generate_sl_non_iterative_template(algo_name, module_name)

def generate_ss_unsupervised_template(algo_name, module_name):
    """SS unsupervised template - Correct SPU-based implementation"""
    return f'''# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Secret Sharing adapter for {algo_name}

{algo_name} is an UNSUPERVISED algorithm.
Data aggregated to SPU with full MPC protection.

Mode: Secret Sharing (SS)
"""

import logging
from typing import Union

try:
    from xlearn.{module_name} import {algo_name}
    USING_XLEARN = True
except ImportError:
    from sklearn.{module_name} import {algo_name}
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import SPU
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class SS{algo_name}:
    """Secret Sharing {algo_name} (Unsupervised)"""

    def __init__(self, spu: SPU, **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")

        self.spu = spu
        self.kwargs = kwargs
        self.model = None
        self._is_fitted = False

        if USING_XLEARN:
            logging.info(f"[SS] SS{algo_name} with JAX acceleration")

    def fit(self, x: Union[FedNdarray, VDataFrame]):
        """Fit (unsupervised - no y needed)"""
        if isinstance(x, VDataFrame):
            x = x.values

        logging.info(f"[SS] SS{algo_name} training in SPU")

        def _spu_fit(X, **kwargs):
            model = {algo_name}(**kwargs)
            model.fit(X)
            return model

        X_spu = x.to(self.spu)
        self.model = self.spu(_spu_fit)(X_spu, **self.kwargs)
        self._is_fitted = True
        return self

    def predict(self, x: Union[FedNdarray, VDataFrame]):
        """Predict cluster labels or anomalies"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        if isinstance(x, VDataFrame):
            x = x.values

        X_spu = x.to(self.spu)
        return self.spu(lambda m, X: m.predict(X))(self.model, X_spu)

    def transform(self, x: Union[FedNdarray, VDataFrame]):
        """Transform data (if supported)"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        if isinstance(x, VDataFrame):
            x = x.values

        X_spu = x.to(self.spu)

        def _transform(m, X):
            if hasattr(m, 'transform'):
                return m.transform(X)
            raise AttributeError("Model does not support transform")

        return self.spu(_transform)(self.model, X_spu)
'''

def generate_ss_non_iterative_template(algo_name, module_name):
    """SS supervised non-iterative template - Correct SPU-based implementation"""
    return f'''# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Secret Sharing adapter for {algo_name}

{algo_name} is a SUPERVISED algorithm.
Data aggregated to SPU with full MPC protection.

Mode: Secret Sharing (SS)
"""

import logging
from typing import Union

try:
    from xlearn.{module_name} import {algo_name}
    USING_XLEARN = True
except ImportError:
    from sklearn.{module_name} import {algo_name}
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import SPU
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class SS{algo_name}:
    """Secret Sharing {algo_name} (Supervised)"""

    def __init__(self, spu: SPU, **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")

        self.spu = spu
        self.kwargs = kwargs
        self.model = None
        self._is_fitted = False

        if USING_XLEARN:
            logging.info(f"[SS] SS{algo_name} with JAX acceleration")

    def fit(self, x: Union[FedNdarray, VDataFrame], y: Union[FedNdarray, VDataFrame]):
        """Fit (supervised - labels required)"""
        if isinstance(x, VDataFrame):
            x = x.values
        if isinstance(y, VDataFrame):
            y = y.values

        logging.info(f"[SS] SS{algo_name} training in SPU")

        def _spu_fit(X, y, **kwargs):
            model = {algo_name}(**kwargs)
            model.fit(X, y)
            return model

        X_spu = x.to(self.spu)
        y_spu = y.to(self.spu)
        self.model = self.spu(_spu_fit)(X_spu, y_spu, **self.kwargs)
        self._is_fitted = True
        return self

    def predict(self, x: Union[FedNdarray, VDataFrame]):
        """Predict using model in SPU"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        if isinstance(x, VDataFrame):
            x = x.values

        X_spu = x.to(self.spu)
        return self.spu(lambda m, X: m.predict(X))(self.model, X_spu)
'''

def generate_ss_iterative_template(algo_name, module_name):
    """SS supervised iterative template - Correct SPU-based implementation"""
    return f'''# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Secret Sharing adapter for {algo_name}

{algo_name} is an ITERATIVE SUPERVISED algorithm.
Data aggregated to SPU with full MPC protection.

Mode: Secret Sharing (SS)
"""

import logging
from typing import Union

try:
    from xlearn.{module_name} import {algo_name}
    USING_XLEARN = True
except ImportError:
    from sklearn.{module_name} import {algo_name}
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import SPU
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class SS{algo_name}:
    """Secret Sharing {algo_name} (Supervised, Iterative)"""

    def __init__(self, spu: SPU, **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")

        self.spu = spu
        self.kwargs = kwargs
        self.model = None
        self._is_fitted = False

        if USING_XLEARN:
            logging.info(f"[SS] SS{algo_name} with JAX acceleration")

    def fit(self, x: Union[FedNdarray, VDataFrame], y: Union[FedNdarray, VDataFrame], epochs: int = 10):
        """Fit (supervised, iterative)"""
        if isinstance(x, VDataFrame):
            x = x.values
        if isinstance(y, VDataFrame):
            y = y.values

        logging.info(f"[SS] SS{algo_name} training in SPU ({{epochs}} epochs)")

        def _spu_fit_iterative(X, y, epochs, **kwargs):
            import numpy as np
            model = {algo_name}(**kwargs)
            for epoch in range(epochs):
                if hasattr(model, 'partial_fit'):
                    if not hasattr(model, 'classes_'):
                        classes = np.unique(y)
                        model.partial_fit(X, y, classes=classes)
                    else:
                        model.partial_fit(X, y)
                else:
                    model.fit(X, y)
            return model

        X_spu = x.to(self.spu)
        y_spu = y.to(self.spu)
        self.model = self.spu(_spu_fit_iterative)(X_spu, y_spu, epochs, **self.kwargs)
        self._is_fitted = True
        return self

    def predict(self, x: Union[FedNdarray, VDataFrame]):
        """Predict using model in SPU"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        if isinstance(x, VDataFrame):
            x = x.values

        X_spu = x.to(self.spu)
        return self.spu(lambda m, X: m.predict(X))(self.model, X_spu)
'''

def generate_sl_unsupervised_template(algo_name, module_name):
    """SL """
    # SL pattern FL 
    return generate_ss_unsupervised_template(algo_name, module_name).replace('SS', 'SL').replace('Secret Sharing', 'Split Learning').replace('SPU', 'PYU')

def generate_sl_non_iterative_template(algo_name, module_name):
    """SL """
    #  FL logic
    return generate_ss_non_iterative_template(algo_name, module_name).replace('SS', 'SL').replace('Secret Sharing', 'Split Learning').replace('SPU', 'PYU')

def generate_sl_iterative_template(algo_name, module_name):
    """SL """
    return generate_ss_iterative_template(algo_name, module_name).replace('SS', 'SL').replace('Secret Sharing', 'Split Learning').replace('SPU', 'PYU')

def main():
    base_path = '/Users/xingqiangchen/jax-sklearn'

    print("="*90)
    print("Batch - ")
    print("="*90)
    print()

    total_created = 0
    total_skipped = 0
    stats_by_type = {}

    for category, info in MISSING_ALGORITHMS.items():
        display_name = info['display_name']
        algorithms = info['algorithms']

        if not algorithms:
            continue

        print(f"\n【{display_name}】 ({len(algorithms)}  × 3 pattern = {len(algorithms) * 3} file)")

        for algo_name in algorithms:
            print(f"\n  : {algo_name}")

            # 
            try:
                char = classify_algorithm(algo_name)
                algo_type = char.get('recommended_implementation', 'unknown')
                print(f"    : {algo_type}")
                print(f"    : {char.get('fit_signature', 'unknown')}")

                # 
                if algo_type not in stats_by_type:
                    stats_by_type[algo_type] = 0
                stats_by_type[algo_type] += 1

            except Exception as e:
                print(f"    ⚠️  failed: {e}")
                algo_type = 'unknown'

            for mode in ['FL', 'SS', 'SL']:
                success, message = generate_algorithm_smart(algo_name, category, mode, base_path)
                if success:
                    print(f"    {mode}: {message}")
                    total_created += 1
                else:
                    print(f"    ⚠️  {mode}: {message}")
                    total_skipped += 1

    print("\n" + "="*90)
    print("completed")
    print("="*90)
    print(f"file: {total_created}")
    print(f"file: {total_skipped}")
    print(f": {total_created + total_skipped}")
    print()

    print(":")
    for algo_type, count in sorted(stats_by_type.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {algo_type}: {count} ")
    print()

if __name__ == '__main__':
    # 
    print("="*90)
    print("⚠️  Batch - ")
    print("="*90)
    print()
    print(":")
    print("  （//）")
    print("  correct")
    print("  correct fit （fit(x) vs fit(x, y) vs fit(x, y, epochs)）")
    print("  （transform/predict/partial_fit）")
    print()
    print(" 234 file:")
    print("  - secretlearn/FL/*/")
    print("  - secretlearn/SS/*/")
    print("  - secretlearn/SL/*/")
    print()

    response = input("？(yes/no): ")
    if response.lower() in ['yes', 'y']:
        main()
    else:
        print("")

