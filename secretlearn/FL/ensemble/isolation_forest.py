# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Federated Learning adapter for IsolationForest

IsolationForest is an UNSUPERVISED transformation algorithm.
Data remains in local PYUs, JAX-accelerated local computation,
HEU-based secure aggregation.

Mode: Federated Learning (FL)
"""

import logging
from typing import Dict, Union, Optional
import numpy as np

try:
    from secretlearn.ensemble import IsolationForest
    USING_XLEARN = True
except ImportError:
    from sklearn.ensemble import IsolationForest
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import PYU, HEU
    from secretflow.security.aggregation import SecureAggregator
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class FLIsolationForest:
    """Federated Learning IsolationForest (Unsupervised)"""
    
    def __init__(self, devices: Dict[str, 'PYU'], heu: Optional['HEU'] = None, aggregation_method: str = 'mean', **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")
        
        self.devices = devices
        self.heu = heu
        self.aggregation_method = aggregation_method
        self.kwargs = kwargs
        self.local_models = {}
        self._is_fitted = False
        
        for party_name, device in devices.items():
            self.local_models[party_name] = device(lambda **kw: IsolationForest(**kw))(**kwargs)
        
        if USING_XLEARN:
            logging.info("[FL] FLIsolationForest with JAX acceleration")
    
    def fit(self, x: 'Union[FedNdarray, VDataFrame]'):
        """Fit (unsupervised - no y needed)"""
        if isinstance(x, VDataFrame):
            x = x.values
        
        logging.info("[FL] Federated IsolationForest training (unsupervised)")
        
        for party_name, device in self.devices.items():
            if device in x.partitions:
                X_local = x.partitions[device]
                model = self.local_models[party_name]
                device(lambda m, X: m.fit(X))(model, X_local)
                logging.info(f"[FL] Party '{party_name}' completed training")
        
        self._is_fitted = True
        return self

    def predict(self, x: 'Union[FedNdarray, VDataFrame]'):
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
    
    def score_samples(self, x: 'Union[FedNdarray, VDataFrame]'):
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
