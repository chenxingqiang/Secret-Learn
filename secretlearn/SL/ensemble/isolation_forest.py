# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Split Learning adapter for IsolationForest

IsolationForest is an UNSUPERVISED algorithm.
Data aggregated to PYU with full MPC protection.

Mode: Split Learning (SL)
"""

import logging
from typing import Union

try:
    from xlearn.ensemble import IsolationForest
    USING_XLEARN = True
except ImportError:
    from sklearn.ensemble import IsolationForest
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import PYU
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class SLIsolationForest:
    """Split Learning IsolationForest (Unsupervised)"""
    
    def __init__(self, spu: PYU, **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")
        self.spu = spu
        self.kwargs = kwargs
        self.model = None
        self._is_fitted = False
        
        if USING_XLEARN:
            logging.info(f"[SL] SLIsolationForest with JAX acceleration")
    
    def fit(self, x: 'Union[FedNdarray, VDataFrame]'):
        """Fit (unsupervised - no y needed)"""
        if isinstance(x, VDataFrame):
            x = x.values
        
        logging.info(f"[SL] SLIsolationForest training in PYU")
        
        def _spu_fit(X, **kwargs):
            model = IsolationForest(**kwargs)
            model.fit(X)
            return model
        
        X_spu = x.to(self.spu)
        self.model = self.spu(_spu_fit)(X_spu, **self.kwargs)
        self._is_fitted = True
        return self
    
    def transform(self, x: 'Union[FedNdarray, VDataFrame]'):
        """Transform data"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        if isinstance(x, VDataFrame):
            x = x.values
        
        X_spu = x.to(self.spu)
        return self.spu(lambda m, X: m.transform(X))(self.model, X_spu)
    
    def predict(self, x: 'Union[FedNdarray, VDataFrame]'):
        """Predict (for clustering/anomaly detection)"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        if isinstance(x, VDataFrame):
            x = x.values
        
        X_spu = x.to(self.spu)
        return self.spu(lambda m, X: m.predict(X))(self.model, X_spu)
