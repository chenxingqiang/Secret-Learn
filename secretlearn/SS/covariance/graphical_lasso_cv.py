# Author: Chen Xingqiang
# SPDX-License-Identifier: BSD-3-Clause

"""
Secret Sharing adapter for GraphicalLassoCV

GraphicalLassoCV is a SUPERVISED non-iterative algorithm.
Data aggregated to SPU with full MPC protection.

Mode: Secret Sharing (SS)
"""

import logging
from typing import Union

try:
    from xlearn.covariance import GraphicalLassoCV
    USING_XLEARN = True
except ImportError:
    from sklearn.covariance import GraphicalLassoCV
    USING_XLEARN = False

try:
    from secretflow.data.ndarray.ndarray import FedNdarray
    from secretflow.data.vertical.dataframe import VDataFrame
    from secretflow.device import SPU
    SECRETFLOW_AVAILABLE = True
except ImportError:
    SECRETFLOW_AVAILABLE = False


class SSGraphicalLassoCV:
    """Secret Sharing GraphicalLassoCV (Supervised, Non-iterative)"""
    
    def __init__(self, spu: 'SPU', **kwargs):
        if not SECRETFLOW_AVAILABLE:
            raise RuntimeError("SecretFlow not installed")
        self.spu = spu
        self.kwargs = kwargs
        self.model = None
        self._is_fitted = False
        
        if USING_XLEARN:
            logging.info(f"[SS] SSGraphicalLassoCV with JAX acceleration")
    
    def fit(self, x: 'Union[FedNdarray, VDataFrame]', y: 'Union[FedNdarray, VDataFrame]'):
        """Fit (supervised, single-pass training in SPU)"""
        if isinstance(x, VDataFrame):
            x = x.values
        if isinstance(y, VDataFrame):
            y = y.values
        
        logging.info(f"[SS] SSGraphicalLassoCV training in SPU")
        
        def _spu_fit(X_parts, y_parts, **kwargs):
            import jax.numpy as jnp
            # Concatenate partitions
            X = jnp.concatenate(X_parts, axis=1) if len(X_parts) > 1 else X_parts[0]
            y = y_parts[0] if isinstance(y_parts, list) else y_parts
            model = GraphicalLassoCV(**kwargs)
            model.fit(X, y)
            return model
        
        # Convert FedNdarray partitions to SPU objects

        
        x_parts = [x.partitions[pyu].to(self.spu) for pyu in x.partitions]

        
        y_parts = [y.partitions[pyu].to(self.spu) for pyu in y.partitions]

        
        

        
        self.model = self.spu(_spu_fit)(x_parts, y_parts, **self.kwargs)
        self._is_fitted = True
        return self
    
    def predict(self, x: 'Union[FedNdarray, VDataFrame]'):
        """Predict using model in SPU"""
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
