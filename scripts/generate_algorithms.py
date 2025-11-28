#!/usr/bin/env python3
"""
智能批量生成算法 - 使用算法分类器和模板生成器

根据算法类型自动选择正确的模板：
- 无监督学习（聚类、降维、预处理）
- 监督学习非迭代
- 监督学习迭代（partial_fit）
"""

import os
import sys
import re

# 添加 secretlearn 到路径
sys.path.insert(0, '/Users/xingqiangchen/jax-sklearn')

from secretlearn.algorithm_classifier import classify_algorithm
from secretlearn.template_generator import generate_template

# 缺失算法列表
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
        'algorithms': ['StandardScaler']  # 这个可能分类错了，应该在 preprocessing
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

# sklearn 模块名映射
MODULE_MAPPING = {
    'linear_models': 'linear_model',
    'clustering': 'cluster',
    'naive_bayes': 'naive_bayes',
    'neural_network': 'neural_network',
}

def camel_to_snake(name):
    """驼峰转下划线"""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def get_sklearn_module_name(category):
    """获取 sklearn 模块名"""
    return MODULE_MAPPING.get(category, category)

def generate_algorithm_smart(algo_name, category, mode, base_path):
    """
    智能生成算法实现 - 使用算法分类器和模板生成器
    
    Parameters
    ----------
    algo_name : str
        算法名称，如 'KMeans'
    category : str
        类别目录名，如 'clustering'
    mode : str
        模式，'FL', 'SS', 或 'SL'
    base_path : str
        基础路径
    """
    # 确定目标目录
    mode_dir = os.path.join(base_path, f'secretlearn/{mode}/{category}')
    os.makedirs(mode_dir, exist_ok=True)
    
    # 生成文件名
    filename = camel_to_snake(algo_name) + '.py'
    filepath = os.path.join(mode_dir, filename)
    
    # 检查文件是否已存在
    if os.path.exists(filepath):
        return False, "已存在"
    
    # 获取 sklearn 模块名
    sklearn_module = get_sklearn_module_name(category)
    
    # 使用算法分类器分类
    try:
        characteristics = classify_algorithm(algo_name)
        template_type = characteristics.get('recommended_implementation', 'supervised_non_iterative')
        
        # 生成模板
        if mode.upper() == 'FL':
            code = generate_template(algo_name, sklearn_module, characteristics, 'fl')
        elif mode.upper() == 'SS':
            code = generate_ss_template_smart(algo_name, sklearn_module, characteristics)
        elif mode.upper() == 'SL':
            code = generate_sl_template_smart(algo_name, sklearn_module, characteristics)
        else:
            return False, f"未知模式: {mode}"
        
        # 写入文件
        with open(filepath, 'w') as f:
            f.write(code)
        
        return True, template_type
        
    except Exception as e:
        return False, f"错误: {str(e)}"

def generate_ss_template_smart(algo_name, module_name, characteristics):
    """生成 SS 模式的智能模板"""
    is_unsupervised = characteristics.get('is_unsupervised', False)
    use_epochs = characteristics.get('use_epochs', False)
    
    if is_unsupervised:
        return generate_ss_unsupervised_template(algo_name, module_name)
    elif use_epochs:
        return generate_ss_iterative_template(algo_name, module_name)
    else:
        return generate_ss_non_iterative_template(algo_name, module_name)

def generate_sl_template_smart(algo_name, module_name, characteristics):
    """生成 SL 模式的智能模板"""
    is_unsupervised = characteristics.get('is_unsupervised', False)
    use_epochs = characteristics.get('use_epochs', False)
    
    if is_unsupervised:
        return generate_sl_unsupervised_template(algo_name, module_name)
    elif use_epochs:
        return generate_sl_iterative_template(algo_name, module_name)
    else:
        return generate_sl_non_iterative_template(algo_name, module_name)

def generate_ss_unsupervised_template(algo_name, module_name):
    """SS 无监督模板"""
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
    
    def transform(self, x: Union[FedNdarray, VDataFrame]):
        """Transform data"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        if isinstance(x, VDataFrame):
            x = x.values
        
        X_spu = x.to(self.spu)
        return self.spu(lambda m, X: m.transform(X))(self.model, X_spu)
    
    def predict(self, x: Union[FedNdarray, VDataFrame]):
        """Predict (for clustering/anomaly detection)"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        if isinstance(x, VDataFrame):
            x = x.values
        
        X_spu = x.to(self.spu)
        return self.spu(lambda m, X: m.predict(X))(self.model, X_spu)
'''

def generate_ss_non_iterative_template(algo_name, module_name):
    """SS 监督学习非迭代模板"""
    return f'''# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Secret Sharing adapter for {algo_name}

{algo_name} is a SUPERVISED non-iterative algorithm.
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
    """Secret Sharing {algo_name} (Supervised, Non-iterative)"""
    
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
        """Fit (supervised, single-pass training in SPU)"""
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
    """SS 监督学习迭代模板"""
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
        """Fit (supervised, iterative training in SPU)"""
        if isinstance(x, VDataFrame):
            x = x.values
        if isinstance(y, VDataFrame):
            y = y.values
        
        logging.info(f"[SS] SS{algo_name} training in SPU ({{epochs}} epochs)")
        
        def _spu_fit_iterative(X, y, epochs, **kwargs):
            model = {algo_name}(**kwargs)
            for epoch in range(epochs):
                if hasattr(model, 'partial_fit'):
                    if not hasattr(model, 'classes_'):
                        import numpy as np
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
    """SL 无监督模板"""
    # SL 模式对于无监督学习与 FL 类似
    return generate_ss_unsupervised_template(algo_name, module_name).replace('SS', 'SL').replace('Secret Sharing', 'Split Learning').replace('SPU', 'PYU')

def generate_sl_non_iterative_template(algo_name, module_name):
    """SL 监督学习非迭代模板"""
    # 基本复用 FL 逻辑
    return generate_ss_non_iterative_template(algo_name, module_name).replace('SS', 'SL').replace('Secret Sharing', 'Split Learning').replace('SPU', 'PYU')

def generate_sl_iterative_template(algo_name, module_name):
    """SL 监督学习迭代模板"""
    return generate_ss_iterative_template(algo_name, module_name).replace('SS', 'SL').replace('Secret Sharing', 'Split Learning').replace('SPU', 'PYU')

def main():
    base_path = '/Users/xingqiangchen/jax-sklearn'
    
    print("="*90)
    print("智能批量生成算法 - 根据算法类型自动选择模板")
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
            
        print(f"\n【{display_name}】 ({len(algorithms)} 个算法 × 3 模式 = {len(algorithms) * 3} 个文件)")
        
        for algo_name in algorithms:
            print(f"\n  算法: {algo_name}")
            
            # 分类算法
            try:
                char = classify_algorithm(algo_name)
                algo_type = char.get('recommended_implementation', 'unknown')
                print(f"    类型: {algo_type}")
                print(f"    签名: {char.get('fit_signature', 'unknown')}")
                
                # 统计
                if algo_type not in stats_by_type:
                    stats_by_type[algo_type] = 0
                stats_by_type[algo_type] += 1
                
            except Exception as e:
                print(f"    ⚠️  分类失败: {e}")
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
    print("生成完成")
    print("="*90)
    print(f"创建文件数: {total_created}")
    print(f"跳过文件数: {total_skipped}")
    print(f"总计: {total_created + total_skipped}")
    print()
    
    print("算法类型统计:")
    for algo_type, count in sorted(stats_by_type.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {algo_type}: {count} 个")
    print()

if __name__ == '__main__':
    # 提示用户确认
    print("="*90)
    print("⚠️  智能批量生成 - 使用算法分类器和模板生成器")
    print("="*90)
    print()
    print("特点:")
    print("  自动识别算法类型（无监督/监督/迭代）")
    print("  根据类型选择正确的模板")
    print("  正确的 fit 签名（fit(x) vs fit(x, y) vs fit(x, y, epochs)）")
    print("  适当的方法（transform/predict/partial_fit）")
    print()
    print("将创建约 234 个文件:")
    print("  - secretlearn/FL/*/")
    print("  - secretlearn/SS/*/")
    print("  - secretlearn/SL/*/")
    print()
    
    response = input("是否继续？(yes/no): ")
    if response.lower() in ['yes', 'y']:
        main()
    else:
        print("操作已取消")

