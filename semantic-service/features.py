"""
Shared feature-extraction helper, kept in its own importable module (not
inline in train.py's __main__ script) for one concrete reason: the trained
sklearn Pipeline pickles a *reference* to this function (via
FunctionTransformer), not its bytecode. If it were defined inside train.py
and train.py were run as `python train.py`, the function's qualified name
at pickle time would be "__main__.select_numeric" -- which resolves fine
while unpickling inside another run of train.py, but fails the moment a
different process (server.py, run as `uvicorn server:app`) tries to load
the same pickle, because ITS __main__ is uvicorn, not train.py. This file
exists so both train.py and server.py import the identical
"features.select_numeric" reference, which pickle can always resolve.
"""
import numpy as np


def select_numeric(df):
    return df[["num_keys", "num_writes"]].values
