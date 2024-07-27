from typing import Optional, Union

import numpy as np
import pandas as pd
import patsy

def rlm(
        formula: str, 
        beta: list, 
        sigma: Union[float, np.ndarray], 
        data: Optional[pd.DataFrame] = None
        ) -> np.ndarray:
    
    """
    Generate a response variable with a linear regression structure.

    Args:
        formula (str): A formula containing the linear predictor.
        beta (list): Vector of regression coefficients.
        sigma (Union[float, np.ndarray]): Error standard deviation. Can be a scalar or a vector with the same length as the number of rows in `data`.
        data (Optional[pd.DataFrame]): A DataFrame containing the covariates entering the linear predictor.

    Returns:
        np.ndarray: A numeric vector containing the generated response variable.

    Raises:
        ValueError: If `data` is None, or if `sigma` is not of the correct size, or if `beta` does not match the number of columns in the design matrix.
    """

    message = "data must be provided."
    if data is None:
        raise ValueError(message)
    
    message = f"sigma must be numeric or of size of the number of rows in data: {data.shape[0]}."
    if np.isscalar(sigma):
        sigma = np.full(data.shape[0], sigma)
    elif len(sigma) != data.shape[0]:
        raise ValueError(message)


    X = patsy.dmatrix(formula, data)
    p = X.shape[1]
    message = "X and beta are incompatible"
    if len(beta) != p:
        raise ValueError(message)
    

    y = X.dot(beta) + np.random.normal(0, sigma)

    return y