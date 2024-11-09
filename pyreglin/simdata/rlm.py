from typing import Optional, Union, Dict, Any
import warnings
import numpy as np
import pandas as pd
import patsy
from numpy.typing import ArrayLike


def rlm(
        formula: str, 
        beta: ArrayLike, 
        sigma: Union[float, np.ndarray], 
        data: Optional[pd.DataFrame] = None,
        contrasts: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None
        ) -> np.ndarray:
    
    """
    Generate a response variable with a linear regression structure.

    Parameters
    ----------
    formula : str
        A formula containing the linear predictor in patsy syntax.
        Example: "y ~ x1 + x2"
    beta : array-like
        Vector of regression coefficients.
    sigma : float or numpy.ndarray
        Error standard deviation. Can be a scalar or a vector with the same 
        length as the number of rows in `data`.
    data : pandas.DataFrame, optional
        A DataFrame containing the covariates entering the linear predictor.
    contrasts : dict, optional
        Dictionary specifying contrasts for categorical variables in `data`.
    random_state : int, optional
        Seed for random number generation. For reproducibility.

    Returns
    -------
    numpy.ndarray
        A numeric vector containing the generated response variable.

    Raises
    ------
    ValueError
        If `data` is None
        If `sigma` is not of the correct size
        If `beta` does not match the number of columns in the design matrix
        If formula is invalid
    TypeError
        If input types are incorrect
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame({
    ...     'x1': np.random.normal(0, 1, 100),
    ...     'x2': np.random.normal(0, 1, 100)
    ... })
    >>> y = rlm("y ~ x1 + x2", beta=[1, 2, 3], sigma=0.5, data=data)

    Notes
    -----
    The function uses patsy for formula processing and numpy for random 
    number generation. The error terms are assumed to be normally distributed.

    See Also
    --------
    patsy.dmatrix : For more information about formula specification
    """
    # Input validation
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
        
    # Convert beta to numpy array for consistency
    beta = np.asarray(beta)
    
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Validate and process sigma
    message = f"sigma must be numeric or of size of the number of rows in data: {data.shape[0]}."
    if np.isscalar(sigma):
        sigma = np.full(data.shape[0], sigma)
    elif len(sigma) != data.shape[0]:
        raise ValueError(message)
    
    try:
        # Generate design matrix
        X = patsy.dmatrix(formula, data, NA_action="raise", contrasts=contrasts)
    except Exception as e:
        raise ValueError(f"Error in formula processing: {str(e)}")
    
    # Validate dimensions
    p = X.shape[1]
    if len(beta) != p:
        raise ValueError(
            f"Dimension mismatch: beta has length {len(beta)}, "
            f"but design matrix has {p} columns"
        )
    
    # Generate response
    y = X.dot(beta) + np.random.normal(0, sigma)
    
    # Warning for potential numerical instability
    if np.any(np.abs(y) > 1e10):
        warnings.warn(
            "Very large response values detected. "
            "Consider scaling your predictors or coefficients.",
            RuntimeWarning
        )
    
    return y