import pandas as pd
from . import sird

def rebuild_cumsum(prev, diff):
    cumsum_0 = diff.cumsum()
    cumsum_1 = pd.DataFrame([prev], index=cumsum_0.index)
    cumsum_1 = cumsum_1.add(cumsum_0, fill_value=0)
    cumsum_1.dropna(inplace=True)
    return cumsum_1
