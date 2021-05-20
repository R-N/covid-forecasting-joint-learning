import pandas as pd
from .kabko import Kabko

"""
The main data loader.
TODO: population & common special dates
"""


class DataLoader:
    def __init__(self, covid=None, psbb=None):
        if covid:
            self.load_covid(covid)
        if psbb:
            self.load_psbb(psbb)

    def load_covid(self, df):
        df = df.rename(columns={
                "infected": "i",
                "recovered": "r",
                "dead": "d"
        })
        df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"])
        df.drop(columns=[df.columns[0], "infected_total"], axis=1, inplace=True)
        self.__covid = df
        return self.__covid

    @property
    def covid(self):
        return self.__covid

    @covid.setter
    def covid(self, value):
        self.load_covid(value)

    def load_psbb(self, df):
        df.loc[:, "start"] = pd.to_datetime(df.loc[:, "start"])
        df.loc[:, "end"] = pd.to_datetime(df.loc[:, "end"])
        self.__psbb = df
        return self.__psbb

    @property
    def psbb(self):
        return self.__psbb

    @psbb.setter
    def psbb(self, value):
        self.load_psbb(value)

    def get_kabko(self, kabko):
        return Kabko(kabko, self)
