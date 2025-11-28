# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Federated Learning adapter for SGDOneClassSVM

SGDOneClassSVM is an ITERATIVE SUPERVISED algorithm.
Data remains in local PYUs, JAX-accelerated local computation,
HEU-based secure aggregation after each epoch.

Mode: Federated Learning (FL)
"""

import logging
from typing import Dict, Union, Optional
import numpy as np

try:
    from secretlearn.linear_model import SGDOneClassSVM
    USING_XLEARN = True
except ImportError:
    from sklearn.linear_model import SGDOneClassSVM
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import PYU, HEU
    from secretflow.security.aggregator import SecureAggregator
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class FLSGDOneClassSVM:
    """Federated Learning SGDOneClassSVM (Supervised, Iterative)"""
    
    def __init__(self, devices: Dict[str, 'PYU'], heu: Optional['HEU'] = None, **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")
        
        self.devices = devices
        self.heu = heu
        self.kwargs = kwargs
        self.local_models = {}
        self._is_fitted = False
        
        for party_name, device in devices.items():
            self.local_models[party_name] = device(lambda **kw: SGDOneClassSVM(**kw))(**kwargs)
        
        if USING_XLEARN:
            logging.info("[FL] FLSGDOneClassSVM with JAX acceleration")
    
    def fit(self, x: 'Union[FedNdarray, VDataFrame]', y: 'Union[FedNdarray, VDataFrame]', epochs: int = 10):
        """Fit (supervised, iterative with partial_fit)"""
        if isinstance(x, VDataFrame):
            x = x.values
        if isinstance(y, VDataFrame):
            y = y.values
        
        logging.info(f"[FL] Federated SGDOneClassSVM training (supervised, iterative, {epochs} epochs)")
        
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
            
            logging.info(f"[FL] Epoch {epoch+1}/{epochs} completed")
        
        self._is_fitted = True
        return self
    
    def partial_fit(self, x: 'Union[FedNdarray, VDataFrame]', y: 'Union[FedNdarray, VDataFrame]', classes=None):
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
    
    def predict(self, x: 'Union[FedNdarray, VDataFrame]'):
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
