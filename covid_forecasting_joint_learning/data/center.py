import numpy as np
import pandas as pd

"""
The main data loader.
TODO: population & common special dates
"""


class DataCenter:
    def __init__(self, covid=None, psbb=None):
        self.__kabko = None
        if covid is not None:
            self.load_covid(covid)
        if psbb is not None:
            self.load_psbb(psbb)

    def load_covid(
        self,
        df,
        kabko_col="kabko",
        date_col="date",
        rename_cols={
            "infected": "i",
            "infectious": "i",
            "recovered": "r",
            "dead": "d"
        },
        drop_cols=["infected_total"],
        drop_first_col=False,
        labels=["i", "r", "d"],
        exclude_kabkos=[
            "AWAK BUAH KAPAL",
            "RS LAPANGAN INDRAPURA"
        ]
    ):
        df = df.rename(columns=rename_cols)
        df.loc[:, date_col] = pd.to_datetime(df[date_col])
        df.loc[:, labels] = df[labels].astype(np.float32)
        drop_cols = [df.columns[0], *drop_cols] if drop_first_col else drop_cols
        df.drop(columns=drop_cols, axis=1, inplace=True)
        df.drop(df.index[df[kabko_col].isin(exclude_kabkos)], inplace=True)
        self.__covid = df
        self.__kabko = df[kabko_col].unique()
        return self.__covid


    @property
    def kabko(self):
        if self.__kabko is None:
            if self.__covid is None:
                raise Exception("Please set/load covid data first")
            self.load_covid(self.__covid)
        return self.__kabko


    @property
    def covid(self):
        return self.__covid

    @covid.setter
    def covid(self, value):
        self.load_covid(value)

    def load_psbb(self, df, start_col="start", end_col="end"):
        df.loc[:, start_col] = pd.to_datetime(df.loc[:, start_col])
        df.loc[:, end_col] = pd.to_datetime(df.loc[:, end_col])
        self.__psbb = df
        return self.__psbb

    @property
    def psbb(self):
        return self.__psbb

    @psbb.setter
    def psbb(self, value):
        self.load_psbb(value)

