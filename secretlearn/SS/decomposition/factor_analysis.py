# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Secret Sharing adapter for FactorAnalysis

FactorAnalysis is an UNSUPERVISED algorithm.
Data aggregated to SPU with full MPC protection.

Mode: Secret Sharing (SS)
"""

import logging
from typing import Union

try:
    from xlearn.decomposition import FactorAnalysis
    USING_XLEARN = True
except ImportError:
    from sklearn.decomposition import FactorAnalysis
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import SPU
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class SSFactorAnalysis:
    """Secret Sharing FactorAnalysis (Unsupervised)"""
    
    def __init__(self, spu: 'SPU', **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")
        
        self.spu = spu
        self.kwargs = kwargs
        self.model = None
        self._is_fitted = False
        
        if USING_XLEARN:
            logging.info(f"[SS] SSFactorAnalysis with JAX acceleration")
    
    def fit(self, x: 'Union[FedNdarray, VDataFrame]'):
        """Fit (unsupervised - no y needed)"""
        if isinstance(x, VDataFrame):
            x = x.values
        
        logging.info(f"[SS] SSFactorAnalysis training in SPU")
        
        def _spu_fit(X, **kwargs):
            model = FactorAnalysis(**kwargs)
            model.fit(X)
            return model
        
        X_spu = x.to(self.spu)
        self.model = self.spu(_spu_fit)(X_spu, **self.kwargs)
        self._is_fitted = True
        return self
    
    def predict(self, x: 'Union[FedNdarray, VDataFrame]'):
        """Predict cluster labels or anomalies"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        if isinstance(x, VDataFrame):
            x = x.values
        
        X_spu = x.to(self.spu)
        return self.spu(lambda m, X: m.predict(X))(self.model, X_spu)
    
    def transform(self, x: 'Union[FedNdarray, VDataFrame]'):
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
