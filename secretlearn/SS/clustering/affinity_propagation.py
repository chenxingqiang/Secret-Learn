# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Secret Sharing adapter for AffinityPropagation

AffinityPropagation is an UNSUPERVISED algorithm.
Data aggregated to SPU with full MPC protection.

Mode: Secret Sharing (SS)
"""

import logging
from typing import Union

try:
    from xlearn.cluster import AffinityPropagation
    USING_XLEARN = True
except ImportError:
    from sklearn.cluster import AffinityPropagation
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import SPU
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class SSAffinityPropagation:
    """Secret Sharing AffinityPropagation (Unsupervised)"""
    
    def __init__(self, spu: 'SPU', **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")
        
        self.spu = spu
        self.kwargs = kwargs
        self.model = None
        self._is_fitted = False
        
        if USING_XLEARN:
            logging.info(f"[SS] SSAffinityPropagation with JAX acceleration")
    
    def fit(self, x: 'Union[FedNdarray, VDataFrame]'):
        """Fit (unsupervised - no y needed)"""
        if isinstance(x, VDataFrame):
            x = x.values
        
        logging.info(f"[SS] SSAffinityPropagation training in SPU")
        
        def _spu_fit(X_parts, **kwargs):
            import jax.numpy as jnp
            # Concatenate partitions
            X = jnp.concatenate(X_parts, axis=1) if len(X_parts) > 1 else X_parts[0]
            model = AffinityPropagation(**kwargs)
            model.fit(X)
            return model
        
        # Convert FedNdarray partitions to SPU objects

        
        x_parts = [x.partitions[pyu].to(self.spu) for pyu in x.partitions]

        
        

        
        self.model = self.spu(_spu_fit)(x_parts, **self.kwargs)
        self._is_fitted = True
        return self
    
    def predict(self, x: 'Union[FedNdarray, VDataFrame]'):
        """Predict cluster labels or anomalies"""
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
