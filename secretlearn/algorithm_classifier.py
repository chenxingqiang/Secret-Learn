# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Algorithm Classifier

Automatically classifies sklearn algorithms by their characteristics:
- Supervised vs Unsupervised
- Iterative vs Non-iterative
- Supports partial_fit or not
- Supports warm_start or not
"""

from typing import Type, Dict, Any
import inspect


class AlgorithmCharacteristics:
    """Algorithm characteristics for proper FL/SL/SS implementation"""
    
    def __init__(self):
        # 无监督学习算法
        self.unsupervised = {
            # 聚类
            'KMeans', 'MiniBatchKMeans', 'DBSCAN', 'AffinityPropagation',
            'AgglomerativeClustering', 'Birch', 'MeanShift', 'SpectralClustering',
            
            # 降维
            'PCA', 'IncrementalPCA', 'KernelPCA', 'TruncatedSVD',
            'NMF', 'MiniBatchNMF', 'FactorAnalysis', 'FastICA',
            'MiniBatchDictionaryLearning',
            
            # 流形学习
            'TSNE', 'Isomap', 'MDS', 'LocallyLinearEmbedding', 'SpectralEmbedding',
            
            # 协方差估计
            'EmpiricalCovariance', 'MinCovDet', 'ShrunkCovariance',
            'LedoitWolf', 'EllipticEnvelope',
            
            # 预处理（无监督）
            'StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler',
            'Normalizer', 'Binarizer', 'QuantileTransformer', 'PowerTransformer',
            'PolynomialFeatures', 'SplineTransformer', 'KBinsDiscretizer',
            
            # 异常检测（无监督）
            'IsolationForest', 'OneClassSVM',
            
            # 特征选择（无监督）
            'VarianceThreshold',
        }
        
        # 支持 partial_fit 的迭代算法
        self.supports_partial_fit = {
            'SGDClassifier', 'SGDRegressor', 'SGDOneClassSVM',
            'PassiveAggressiveClassifier', 'PassiveAggressiveRegressor',
            'Perceptron',
            'GaussianNB', 'MultinomialNB', 'BernoulliNB',
            'CategoricalNB', 'ComplementNB',
            'MiniBatchKMeans', 'MiniBatchNMF',
            'IncrementalPCA', 'MiniBatchDictionaryLearning',
        }
        
        # 支持 warm_start 的算法
        self.supports_warm_start = {
            'LogisticRegression', 'LogisticRegressionCV',
            'MLPClassifier', 'MLPRegressor',
            'RandomForestClassifier', 'RandomForestRegressor',
            'ExtraTreesClassifier', 'ExtraTreesRegressor',
            'GradientBoostingClassifier', 'GradientBoostingRegressor',
            'HistGradientBoostingClassifier', 'HistGradientBoostingRegressor',
        }
        
        # 迭代式神经网络（使用 warm_start 而非 partial_fit）
        self.iterative_neural_network = {
            'MLPClassifier', 'MLPRegressor',
        }
        
        # Lazy learning（无需训练循环）
        self.lazy_learning = {
            'KNeighborsClassifier', 'KNeighborsRegressor',
            'RadiusNeighborsClassifier', 'RadiusNeighborsRegressor',
            'NearestCentroid', 'LocalOutlierFactor',
        }
        
        # 需要特殊处理的算法
        self.special_handling = {
            'LabelPropagation', 'LabelSpreading',  # 半监督
            'CalibratedClassifierCV',  # 元估计器
            'OneVsOneClassifier', 'OneVsRestClassifier',  # 元估计器
            'MultiOutputClassifier', 'MultiOutputRegressor',  # 元估计器
            'DummyClassifier', 'DummyRegressor',  # 虚拟估计器
        }
    
    def classify(self, algorithm_name: str) -> Dict[str, Any]:
        """
        Classify an algorithm by its characteristics
        
        Returns
        -------
        characteristics : dict
            Dictionary with:
            - is_unsupervised: bool
            - supports_partial_fit: bool
            - supports_warm_start: bool
            - is_iterative: bool
            - is_lazy_learning: bool
            - needs_special_handling: bool
            - recommended_implementation: str
        """
        result = {
            'is_unsupervised': algorithm_name in self.unsupervised,
            'supports_partial_fit': algorithm_name in self.supports_partial_fit,
            'supports_warm_start': algorithm_name in self.supports_warm_start,
            'is_iterative_neural': algorithm_name in self.iterative_neural_network,
            'is_lazy_learning': algorithm_name in self.lazy_learning,
            'needs_special_handling': algorithm_name in self.special_handling,
        }
        
        # 推荐的实现方式
        if result['is_unsupervised']:
            result['recommended_implementation'] = 'unsupervised'
            result['fit_signature'] = 'fit(x)'
            result['use_epochs'] = False
            result['use_warm_start'] = False
        elif result['supports_partial_fit'] and not result['is_iterative_neural']:
            result['recommended_implementation'] = 'iterative_partial_fit'
            result['fit_signature'] = 'fit(x, y, epochs)'
            result['use_epochs'] = True
            result['use_warm_start'] = False
        elif result['is_iterative_neural']:
            result['recommended_implementation'] = 'iterative_neural'
            result['fit_signature'] = 'fit(x, y, epochs)'
            result['use_epochs'] = True
            result['use_warm_start'] = True
        elif result['is_lazy_learning']:
            result['recommended_implementation'] = 'lazy_learning'
            result['fit_signature'] = 'fit(x, y)'
            result['use_epochs'] = False
            result['use_warm_start'] = False
        else:
            result['recommended_implementation'] = 'non_iterative_supervised'
            result['fit_signature'] = 'fit(x, y)'
            result['use_epochs'] = False
            result['use_warm_start'] = False
        
        return result
    
    def get_template_type(self, algorithm_name: str) -> str:
        """
        Get the template type for an algorithm
        
        Returns
        -------
        template_type : str
            One of: 'unsupervised', 'supervised_iterative', 'supervised_non_iterative'
        """
        char = self.classify(algorithm_name)
        
        if char['is_unsupervised']:
            return 'unsupervised'
        elif char['use_epochs']:
            return 'supervised_iterative'
        else:
            return 'supervised_non_iterative'


# Global instance
algorithm_classifier = AlgorithmCharacteristics()


def classify_algorithm(algorithm_name: str) -> Dict[str, Any]:
    """
    Classify an algorithm by name
    
    Examples
    --------
    >>> char = classify_algorithm('KMeans')
    >>> print(char['is_unsupervised'])  # True
    >>> print(char['fit_signature'])   # 'fit(x)'
    
    >>> char = classify_algorithm('SGDClassifier')
    >>> print(char['supports_partial_fit'])  # True
    >>> print(char['use_epochs'])  # True
    """
    return algorithm_classifier.classify(algorithm_name)


def classify_algorithm_from_class(sklearn_class: Type) -> Dict[str, Any]:
    """
    Classify an algorithm by its class
    
    Examples
    --------
    >>> from sklearn.cluster import KMeans
    >>> char = classify_algorithm_from_class(KMeans)
    >>> print(char['is_unsupervised'])  # True
    """
    algorithm_name = sklearn_class.__name__
    char = algorithm_classifier.classify(algorithm_name)
    
    # 如果没有在预定义列表中，尝试从类本身推断
    if not any([
        char['is_unsupervised'],
        char['supports_partial_fit'],
        char['supports_warm_start'],
        char['is_lazy_learning'],
    ]):
        # 检查方法签名
        try:
            fit_sig = inspect.signature(sklearn_class.fit)
            params = list(fit_sig.parameters.keys())
            
            # 检查是否有 y 参数
            has_y = 'y' in params
            
            if not has_y:
                char['is_unsupervised'] = True
                char['fit_signature'] = 'fit(x)'
            else:
                char['fit_signature'] = 'fit(x, y)'
            
            # 检查是否有 partial_fit
            if hasattr(sklearn_class, 'partial_fit'):
                char['supports_partial_fit'] = True
                char['use_epochs'] = True
        except:
            pass
    
    return char


if __name__ == "__main__":
    """Test classification"""
    print("Algorithm Classification Examples")
    print("=" * 70)
    
    test_algorithms = [
        'KMeans', 'PCA', 'IsolationForest',  # Unsupervised
        'LinearRegression', 'SVC', 'RandomForestClassifier',  # Supervised non-iterative
        'SGDClassifier', 'MLPClassifier', 'GaussianNB',  # Supervised iterative
    ]
    
    for algo in test_algorithms:
        char = classify_algorithm(algo)
        print(f"\n{algo}:")
        print(f"  - Type: {char['recommended_implementation']}")
        print(f"  - Fit signature: {char['fit_signature']}")
        print(f"  - Use epochs: {char['use_epochs']}")
        print(f"  - Use warm_start: {char['use_warm_start']}")

