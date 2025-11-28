# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Secret Sharing adapter for OneClassSVM

OneClassSVM is an UNSUPERVISED algorithm.
Data aggregated to SPU with full MPC protection.

Mode: Secret Sharing (SS)
"""

import logging
from typing import Union

try:
    from xlearn.svm import OneClassSVM
    USING_XLEARN = True
except ImportError:
    from sklearn.svm import OneClassSVM
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import SPU
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class SSOneClassSVM:
    """Secret Sharing OneClassSVM (Unsupervised)"""
    
    def __init__(self, spu: SPU, **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")
        self.spu = spu
        self.kwargs = kwargs
        self.model = None
        self._is_fitted = False
        
        if USING_XLEARN:
            logging.info(f"[SS] SSOneClassSVM with JAX acceleration")
    
    def fit(self, x: Union[FedNdarray, VDataFrame]):
        """Fit (unsupervised - no y needed)"""
        if isinstance(x, VDataFrame):
            x = x.values
        
        logging.info(f"[SS] SSOneClassSVM training in SPU")
        
        def _spu_fit(X, **kwargs):
            model = OneClassSVM(**kwargs)
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
