# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Secret Sharing adapter for PassiveAggressiveClassifier

PassiveAggressiveClassifier is an ITERATIVE SUPERVISED algorithm.
Data aggregated to SPU with full MPC protection.

Mode: Secret Sharing (SS)
"""

import logging
from typing import Union

try:
    from xlearn.linear_model import PassiveAggressiveClassifier
    USING_XLEARN = True
except ImportError:
    from sklearn.linear_model import PassiveAggressiveClassifier
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import SPU
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class SSPassiveAggressiveClassifier:
    """Secret Sharing PassiveAggressiveClassifier (Supervised, Iterative)"""
    
    def __init__(self, spu: 'SPU', **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")
        
        self.spu = spu
        self.kwargs = kwargs
        self.model = None
        self._is_fitted = False
        
        if USING_XLEARN:
            logging.info(f"[SS] SSPassiveAggressiveClassifier with JAX acceleration")
    
    def fit(self, x: 'Union[FedNdarray, VDataFrame]', y: 'Union[FedNdarray, VDataFrame]', epochs: int = 10):
        """Fit (supervised, iterative)"""
        if isinstance(x, VDataFrame):
            x = x.values
        if isinstance(y, VDataFrame):
            y = y.values
        
        logging.info(f"[SS] SSPassiveAggressiveClassifier training in SPU ({epochs} epochs)")
        
        def _spu_fit_iterative(X, y, epochs, **kwargs):
            import numpy as np
            model = PassiveAggressiveClassifier(**kwargs)
            for epoch in range(epochs):
                if hasattr(model, 'partial_fit'):
                    if not hasattr(model, 'classes_'):
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
    
    def predict(self, x: 'Union[FedNdarray, VDataFrame]'):
        """Predict using model in SPU"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        if isinstance(x, VDataFrame):
            x = x.values
        
        X_spu = x.to(self.spu)
        return self.spu(lambda m, X: m.predict(X))(self.model, X_spu)
