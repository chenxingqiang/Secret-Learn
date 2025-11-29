# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Secret Sharing adapter for IsolationForest

IsolationForest is an UNSUPERVISED algorithm.
Data aggregated to SPU with full MPC protection.

Mode: Secret Sharing (SS)
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
    from secretflow.device import SPU
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class SSIsolationForest:
    """Secret Sharing IsolationForest (Unsupervised)"""
    
    def __init__(self, spu: 'SPU', **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")
        self.spu = spu
        self.kwargs = kwargs
        self.model = None
        self._is_fitted = False
        
        if USING_XLEARN:
            logging.info(f"[SS] SSIsolationForest with JAX acceleration")
    
    def fit(self, x: 'Union[FedNdarray, VDataFrame]'):
        """Fit (unsupervised - no y needed)"""
        if isinstance(x, VDataFrame):
            x = x.values
        
        logging.info(f"[SS] SSIsolationForest training in SPU")
        
        def _spu_fit(X_parts, **kwargs):
            import jax.numpy as jnp
            # Concatenate partitions
            X = jnp.concatenate(X_parts, axis=1) if len(X_parts) > 1 else X_parts[0]
            model = IsolationForest(**kwargs)
            model.fit(X)
            return model
        
        # Convert FedNdarray partitions to SPU objects

        
        x_parts = [x.partitions[pyu].to(self.spu) for pyu in x.partitions]

        
        

        
        self.model = self.spu(_spu_fit)(x_parts, **self.kwargs)
        self._is_fitted = True
        return self
    
    def transform(self, x: 'Union[FedNdarray, VDataFrame]'):
        """Transform data"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        if isinstance(x, VDataFrame):
            x = x.values
        
        # Convert FedNdarray partitions to SPU

        
        x_parts = [x.partitions[pyu].to(self.spu) for pyu in x.partitions]

        
        

        
        def _spu_transform(m, X_parts):

        
            import jax.numpy as jnp

        
            X = jnp.concatenate(X_parts, axis=1) if len(X_parts) > 1 else X_parts[0]

        
            return m.transform(X)

        
        

        
        return self.spu(_spu_transform)(self.model, x_parts)
    
    def predict(self, x: 'Union[FedNdarray, VDataFrame]'):
        """Predict (for clustering/anomaly detection)"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        if isinstance(x, VDataFrame):
            x = x.values
        
        # Convert FedNdarray partitions to SPU

        
        x_parts = [x.partitions[pyu].to(self.spu) for pyu in x.partitions]

        
        

        
        def _spu_predict(m, X_parts):

        
            import jax.numpy as jnp

        
            X = jnp.concatenate(X_parts, axis=1) if len(X_parts) > 1 else X_parts[0]

        
            return m.predict(X)

        
        

        
        return self.spu(_spu_predict)(self.model, x_parts)
