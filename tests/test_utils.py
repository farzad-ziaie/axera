import numpy as np
from axera.utils import impute_and_drop


def test_impute_and_drop():
    import pandas as pd
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    result = impute_and_drop(df, impute_cols=["a"], drop_cols=["c"])
    assert "c" not in result.columns
    assert not result["a"].isna().any()
