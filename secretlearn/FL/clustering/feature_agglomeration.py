# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Federated Learning adapter for FeatureAgglomeration

FeatureAgglomeration is a SUPERVISED non-iterative algorithm.
Data remains in local PYUs, JAX-accelerated local computation,
HEU-based secure aggregation.

Mode: Federated Learning (FL)
"""

import logging
from typing import Dict, Union, Optional
import numpy as np

try:
    from secretlearn.cluster import FeatureAgglomeration
    USING_XLEARN = True
except ImportError:
    from sklearn.cluster import FeatureAgglomeration
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import PYU, HEU
    from secretflow.security.aggregation import SecureAggregator
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class FLFeatureAgglomeration:
    """Federated Learning FeatureAgglomeration (Supervised, Non-iterative)"""
    
    def __init__(self, devices: Dict[str, 'PYU'], heu: Optional['HEU'] = None, **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")
        
        self.devices = devices
        self.heu = heu
        self.kwargs = kwargs
        self.local_models = {}
        self._is_fitted = False
        
        for party_name, device in devices.items():
            self.local_models[party_name] = device(lambda **kw: FeatureAgglomeration(**kw))(**kwargs)
        
        if USING_XLEARN:
            logging.info("[FL] FLFeatureAgglomeration with JAX acceleration")
    
    def fit(self, x: 'Union[FedNdarray, VDataFrame]', y: 'Union[FedNdarray, VDataFrame]'):
        """Fit (supervised, single-pass training)"""
        if isinstance(x, VDataFrame):
            x = x.values
        if isinstance(y, VDataFrame):
            y = y.values
        
        logging.info("[FL] Federated FeatureAgglomeration training (supervised, non-iterative)")
        
        for party_name, device in self.devices.items():
            if device in x.partitions:
                X_local = x.partitions[device]
                y_local = y.partitions.get(device, y)
                model = self.local_models[party_name]
                device(lambda m, X, y: m.fit(X, y))(model, X_local, y_local)
                logging.info(f"[FL] Party '{party_name}' completed training")
        
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
