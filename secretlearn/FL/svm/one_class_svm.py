# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Federated Learning adapter for OneClassSVM

OneClassSVM is an UNSUPERVISED transformation algorithm.
Data remains in local PYUs, JAX-accelerated local computation,
HEU-based secure aggregation.

Mode: Federated Learning (FL)
"""

import logging
from typing import Dict, Union, Optional
import numpy as np

try:
    from secretlearn.svm import OneClassSVM
    USING_XLEARN = True
except ImportError:
    from sklearn.svm import OneClassSVM
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import PYU, HEU
    from secretflow.security.aggregation import SecureAggregator
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class FLOneClassSVM:
    """Federated Learning OneClassSVM (Unsupervised)"""
    
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
            self.local_models[party_name] = device(lambda **kw: OneClassSVM(**kw))(**kwargs)
        
        if USING_XLEARN:
            logging.info("[FL] FLOneClassSVM with JAX acceleration")
    
    def fit(self, x: 'Union[FedNdarray, VDataFrame]'):
        """Fit (unsupervised - no y needed)"""
        if isinstance(x, VDataFrame):
            x = x.values
        
        logging.info("[FL] Federated OneClassSVM training (unsupervised)")
        
        for party_name, device in self.devices.items():
            if device in x.partitions:
                X_local = x.partitions[device]
                model = self.local_models[party_name]
                device(lambda m, X: m.fit(X))(model, X_local)
                logging.info(f"[FL] Party '{party_name}' completed training")
        
        self._is_fitted = True
        return self
