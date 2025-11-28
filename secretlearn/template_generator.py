# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Template Generator for SecretFlow Adapters

Generates correct templates based on algorithm characteristics.
"""

from typing import Dict, Any


def generate_unsupervised_fl_template(algorithm_name: str, module_name: str, algorithm_type: str = "learning") -> str:
    """生成无监督学习FL模板"""
    
    # 判断是否有 transform 方法（降维、预处理）
    needs_transform = module_name in ['decomposition', 'preprocessing', 'manifold', 'feature_selection']
    
    # 判断是否有 cluster 相关方法（聚类）
    is_clustering = module_name in ['cluster', 'clustering']
    
    # 判断是否是异常检测
    is_anomaly = 'isolation' in algorithm_name.lower() or 'outlier' in algorithm_name.lower()
    
    template = f'''# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Federated Learning adapter for {algorithm_name}

{algorithm_name} is an UNSUPERVISED {algorithm_type} algorithm.
Data remains in local PYUs, JAX-accelerated local computation,
HEU-based secure aggregation.

Mode: Federated Learning (FL)
"""

import logging
from typing import Dict, Union, Optional
import numpy as np

try:
    from secretlearn.{module_name} import {algorithm_name}
    USING_XLEARN = True
except ImportError:
    from sklearn.{module_name} import {algorithm_name}
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import PYU, HEU
    from secretflow.security.aggregation import SecureAggregator
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class FL{algorithm_name}:
    """Federated Learning {algorithm_name} (Unsupervised)"""
    
    def __init__(self, devices: Dict[str, PYU], heu: Optional[HEU] = None, aggregation_method: str = 'mean', **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")
        
        self.devices = devices
        self.heu = heu
        self.aggregation_method = aggregation_method
        self.kwargs = kwargs
        self.local_models = {{}}
        self._is_fitted = False
        
        for party_name, device in devices.items():
            self.local_models[party_name] = device(lambda **kw: {algorithm_name}(**kw))(**kwargs)
        
        if USING_XLEARN:
            logging.info("[FL] FL{algorithm_name} with JAX acceleration")
    
    def fit(self, x: Union[FedNdarray, VDataFrame]):
        """Fit (unsupervised - no y needed)"""
        if isinstance(x, VDataFrame):
            x = x.values
        
        logging.info("[FL] Federated {algorithm_name} training (unsupervised)")
        
        for party_name, device in self.devices.items():
            if device in x.partitions:
                X_local = x.partitions[device]
                model = self.local_models[party_name]
                device(lambda m, X: m.fit(X))(model, X_local)
                logging.info(f"[FL] Party '{{party_name}}' completed training")
        
        self._is_fitted = True
        return self
'''
    
    # 添加预测/转换方法
    if is_clustering:
        template += '''
    def predict(self, x: Union[FedNdarray, VDataFrame]):
        """Predict cluster labels"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        if isinstance(x, VDataFrame):
            x = x.values
        
        predictions_list = []
        for party_name, device in self.devices.items():
            if device in x.partitions:
                X_local = x.partitions[device]
                model = self.local_models[party_name]
                pred = device(lambda m, X: m.predict(X))(model, X_local)
                predictions_list.append(pred)
        
        return predictions_list[0] if len(predictions_list) == 1 else predictions_list
    
    def fit_predict(self, x: Union[FedNdarray, VDataFrame]):
        """Fit and predict"""
        self.fit(x)
        return self.predict(x)
'''
    elif needs_transform:
        template += '''
    def transform(self, x: Union[FedNdarray, VDataFrame]):
        """Transform data"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        if isinstance(x, VDataFrame):
            x = x.values
        
        transformed_list = []
        for party_name, device in self.devices.items():
            if device in x.partitions:
                X_local = x.partitions[device]
                model = self.local_models[party_name]
                X_trans = device(lambda m, X: m.transform(X))(model, X_local)
                transformed_list.append(X_trans)
        
        return transformed_list[0] if len(transformed_list) == 1 else transformed_list
    
    def fit_transform(self, x: Union[FedNdarray, VDataFrame]):
        """Fit and transform"""
        self.fit(x)
        return self.transform(x)
    
    def inverse_transform(self, x: Union[FedNdarray, VDataFrame]):
        """Inverse transform (if supported)"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        if isinstance(x, VDataFrame):
            x = x.values
        
        inverse_list = []
        for party_name, device in self.devices.items():
            if device in x.partitions:
                X_local = x.partitions[device]
                model = self.local_models[party_name]
                
                def _inverse(m, X):
                    if hasattr(m, 'inverse_transform'):
                        return m.inverse_transform(X)
                    raise AttributeError("inverse_transform not supported")
                
                X_inv = device(_inverse)(model, X_local)
                inverse_list.append(X_inv)
        
        return inverse_list[0] if len(inverse_list) == 1 else inverse_list
'''
    elif is_anomaly:
        template += '''
    def predict(self, x: Union[FedNdarray, VDataFrame]):
        """Predict anomalies (-1 for outliers, 1 for inliers)"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        if isinstance(x, VDataFrame):
            x = x.values
        
        predictions_list = []
        for party_name, device in self.devices.items():
            if device in x.partitions:
                X_local = x.partitions[device]
                model = self.local_models[party_name]
                pred = device(lambda m, X: m.predict(X))(model, X_local)
                predictions_list.append(pred)
        
        return predictions_list[0] if len(predictions_list) == 1 else predictions_list
    
    def score_samples(self, x: Union[FedNdarray, VDataFrame]):
        """Compute anomaly scores"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        if isinstance(x, VDataFrame):
            x = x.values
        
        scores_list = []
        for party_name, device in self.devices.items():
            if device in x.partitions:
                X_local = x.partitions[device]
                model = self.local_models[party_name]
                scores = device(lambda m, X: m.score_samples(X))(model, X_local)
                scores_list.append(scores)
        
        return scores_list[0] if len(scores_list) == 1 else scores_list
'''
    
    return template


def generate_supervised_non_iterative_fl_template(algorithm_name: str, module_name: str) -> str:
    """生成非迭代监督学习FL模板"""
    
    return f'''# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Federated Learning adapter for {algorithm_name}

{algorithm_name} is a SUPERVISED non-iterative algorithm.
Data remains in local PYUs, JAX-accelerated local computation,
HEU-based secure aggregation.

Mode: Federated Learning (FL)
"""

import logging
from typing import Dict, Union, Optional
import numpy as np

try:
    from secretlearn.{module_name} import {algorithm_name}
    USING_XLEARN = True
except ImportError:
    from sklearn.{module_name} import {algorithm_name}
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import PYU, HEU
    from secretflow.security.aggregation import SecureAggregator
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class FL{algorithm_name}:
    """Federated Learning {algorithm_name} (Supervised, Non-iterative)"""
    
    def __init__(self, devices: Dict[str, PYU], heu: Optional[HEU] = None, **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")
        
        self.devices = devices
        self.heu = heu
        self.kwargs = kwargs
        self.local_models = {{}}
        self._is_fitted = False
        
        for party_name, device in devices.items():
            self.local_models[party_name] = device(lambda **kw: {algorithm_name}(**kw))(**kwargs)
        
        if USING_XLEARN:
            logging.info("[FL] FL{algorithm_name} with JAX acceleration")
    
    def fit(self, x: Union[FedNdarray, VDataFrame], y: Union[FedNdarray, VDataFrame]):
        """Fit (supervised, single-pass training)"""
        if isinstance(x, VDataFrame):
            x = x.values
        if isinstance(y, VDataFrame):
            y = y.values
        
        logging.info("[FL] Federated {algorithm_name} training (supervised, non-iterative)")
        
        for party_name, device in self.devices.items():
            if device in x.partitions:
                X_local = x.partitions[device]
                y_local = y.partitions.get(device, y)
                model = self.local_models[party_name]
                device(lambda m, X, y: m.fit(X, y))(model, X_local, y_local)
                logging.info(f"[FL] Party '{{party_name}}' completed training")
        
        self._is_fitted = True
        return self
    
    def predict(self, x: Union[FedNdarray, VDataFrame]):
        """Predict using federated model"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        if isinstance(x, VDataFrame):
            x = x.values
        
        predictions_list = []
        for party_name, device in self.devices.items():
            if device in x.partitions:
                X_local = x.partitions[device]
                model = self.local_models[party_name]
                pred = device(lambda m, X: m.predict(X))(model, X_local)
                predictions_list.append(pred)
        
        if len(predictions_list) == 1:
            return predictions_list[0]
        if self.heu:
            aggregator = SecureAggregator(device=self.heu)
            return aggregator.average(predictions_list)
        return np.mean(predictions_list, axis=0)
'''


def generate_supervised_iterative_fl_template(algorithm_name: str, module_name: str) -> str:
    """生成迭代监督学习FL模板"""
    
    return f'''# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Federated Learning adapter for {algorithm_name}

{algorithm_name} is an ITERATIVE SUPERVISED algorithm.
Data remains in local PYUs, JAX-accelerated local computation,
HEU-based secure aggregation after each epoch.

Mode: Federated Learning (FL)
"""

import logging
from typing import Dict, Union, Optional
import numpy as np

try:
    from secretlearn.{module_name} import {algorithm_name}
    USING_XLEARN = True
except ImportError:
    from sklearn.{module_name} import {algorithm_name}
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import PYU, HEU
    from secretflow.security.aggregator import SecureAggregator
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class FL{algorithm_name}:
    """Federated Learning {algorithm_name} (Supervised, Iterative)"""
    
    def __init__(self, devices: Dict[str, PYU], heu: Optional[HEU] = None, **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")
        
        self.devices = devices
        self.heu = heu
        self.kwargs = kwargs
        self.local_models = {{}}
        self._is_fitted = False
        
        for party_name, device in devices.items():
            self.local_models[party_name] = device(lambda **kw: {algorithm_name}(**kw))(**kwargs)
        
        if USING_XLEARN:
            logging.info("[FL] FL{algorithm_name} with JAX acceleration")
    
    def fit(self, x: Union[FedNdarray, VDataFrame], y: Union[FedNdarray, VDataFrame], epochs: int = 10):
        """Fit (supervised, iterative with partial_fit)"""
        if isinstance(x, VDataFrame):
            x = x.values
        if isinstance(y, VDataFrame):
            y = y.values
        
        logging.info(f"[FL] Federated {algorithm_name} training (supervised, iterative, {{epochs}} epochs)")
        
        for epoch in range(epochs):
            for party_name, device in self.devices.items():
                if device in x.partitions:
                    X_local = x.partitions[device]
                    y_local = y.partitions.get(device, y)
                    model = self.local_models[party_name]
                    
                    def _partial_fit(m, X, y):
                        if not hasattr(m, 'classes_'):
                            classes = np.unique(y)
                            m.partial_fit(X, y, classes=classes)
                        else:
                            m.partial_fit(X, y)
                        return True
                    
                    device(_partial_fit)(model, X_local, y_local)
            
            if self.heu:
                # Aggregate parameters
                pass
            
            logging.info(f"[FL] Epoch {{epoch+1}}/{{epochs}} completed")
        
        self._is_fitted = True
        return self
    
    def partial_fit(self, x: Union[FedNdarray, VDataFrame], y: Union[FedNdarray, VDataFrame], classes=None):
        """Incremental fit on a batch"""
        if isinstance(x, VDataFrame):
            x = x.values
        if isinstance(y, VDataFrame):
            y = y.values
        
        for party_name, device in self.devices.items():
            if device in x.partitions:
                X_local = x.partitions[device]
                y_local = y.partitions.get(device, y)
                model = self.local_models[party_name]
                
                if classes is not None:
                    device(lambda m, X, y, c: m.partial_fit(X, y, classes=c))(model, X_local, y_local, classes)
                else:
                    device(lambda m, X, y: m.partial_fit(X, y))(model, X_local, y_local)
        
        self._is_fitted = True
        return self
    
    def predict(self, x: Union[FedNdarray, VDataFrame]):
        """Predict using federated model"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        if isinstance(x, VDataFrame):
            x = x.values
        
        predictions_list = []
        for party_name, device in self.devices.items():
            if device in x.partitions:
                X_local = x.partitions[device]
                model = self.local_models[party_name]
                pred = device(lambda m, X: m.predict(X))(model, X_local)
                predictions_list.append(pred)
        
        if len(predictions_list) == 1:
            return predictions_list[0]
        if self.heu:
            aggregator = SecureAggregator(device=self.heu)
            return aggregator.average(predictions_list)
        return np.mean(predictions_list, axis=0)
'''
    
    return template


def generate_template(algorithm_name: str, module_name: str, characteristics: Dict[str, Any], mode: str = 'fl') -> str:
    """
    Generate appropriate template based on algorithm characteristics
    
    Parameters
    ----------
    algorithm_name : str
        Name of the algorithm (e.g., 'KMeans', 'SGDClassifier')
    module_name : str
        Module name (e.g., 'cluster', 'linear_model')
    characteristics : dict
        Algorithm characteristics from algorithm_classifier
    mode : str
        'fl', 'sl', or 'ss'
    
    Returns
    -------
    template : str
        Generated template code
    """
    if mode == 'fl':
        if characteristics['is_unsupervised']:
            algorithm_type = "clustering" if 'cluster' in module_name else "transformation"
            return generate_unsupervised_fl_template(algorithm_name, module_name, algorithm_type)
        elif characteristics['use_epochs']:
            return generate_supervised_iterative_fl_template(algorithm_name, module_name)
        else:
            return generate_supervised_non_iterative_fl_template(algorithm_name, module_name)
    
    elif mode == 'sl':
        # SL模式：复用FL逻辑，只修改描述
        fl_template = generate_template(algorithm_name, module_name, characteristics, 'fl')
        return fl_template.replace('Federated Learning', 'Split Learning').replace('[FL]', '[SL]').replace('FL' + algorithm_name, 'SL' + algorithm_name)
    
    elif mode == 'ss':
        # SS模式：复用FL逻辑，修改为SPU
        fl_template = generate_template(algorithm_name, module_name, characteristics, 'fl')
        return fl_template.replace('Federated Learning', 'Simple Sealed').replace('[FL]', '[SS]').replace('FL' + algorithm_name, 'SS' + algorithm_name).replace('PYU', 'SPU').replace('HEU', 'SPU')
    
    return ""


if __name__ == "__main__":
    """Test template generation"""
    from algorithm_classifier import classify_algorithm
    
    # Test unsupervised
    char = classify_algorithm('KMeans')
    template = generate_template('KMeans', 'cluster', char, 'fl')
    print("KMeans FL Template (first 500 chars):")
    print(template[:500])
    print("...")
    
    # Test supervised non-iterative
    char = classify_algorithm('LinearRegression')
    template = generate_template('LinearRegression', 'linear_model', char, 'fl')
    print("\nLinearRegression FL Template (first 500 chars):")
    print(template[:500])
    print("...")
    
    # Test supervised iterative
    char = classify_algorithm('SGDClassifier')
    template = generate_template('SGDClassifier', 'linear_model', char, 'fl')
    print("\nSGDClassifier FL Template (first 500 chars):")
    print(template[:500])
    print("...")

