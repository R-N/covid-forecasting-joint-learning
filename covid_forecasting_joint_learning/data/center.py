import numpy as np
import pandas as pd
from . import util as DataUtil
from . import cols as DataCol

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
        self.__dates_global = pd.DataFrame([], columns=[
            DataCol.NAME,
            DataCol.START,
            DataCol.END,
            DataCol.VAL
        ])
        self.__dates_local = pd.DataFrame([], columns=[
            DataCol.KABKO,
            DataCol.NAME,
            DataCol.START,
            DataCol.END,
            DataCol.VAL
        ])

    def load_covid_local(
        self,
        df,
        kabko_col="kabko",
        date_col="date",
        rename_cols={
            "infected": DataCol.I,
            "infectious": DataCol.I,
            "recovered": DataCol.R,
            "dead": DataCol.D
        },
        drop_cols=["infected_total"],
        drop_first_col=False,
        exclude_kabkos=[
            "AWAK BUAH KAPAL",
            "RS LAPANGAN INDRAPURA"
        ]
    ):
        df = df.copy()
        labels = [DataCol.I, DataCol.R, DataCol.D]
        df.loc[:, date_col] = pd.to_datetime(df[date_col])
        drop_cols = [df.columns[0], *drop_cols] if drop_first_col else drop_cols
        df.drop(columns=drop_cols, axis=1, inplace=True)
        df.drop(df.index[df[kabko_col].isin(exclude_kabkos)], inplace=True)
        rename_cols = {
            kabko_col: DataCol.KABKO,
            date_col: DataCol.DATE,
            **rename_cols
        }
        df.rename(columns=rename_cols, inplace=True)
        df.loc[:, labels] = df[labels].astype(np.float32)
        self.__covid_local = df
        self.__kabko = df[kabko_col].unique()
        return self.__covid_local

    @property
    def kabko(self):
        if self.__kabko is None:
            if self.__covid_local is None:
                raise Exception("Please set/load covid data first")
            self.load_covid_local(self.__covid_local)
        return self.__kabko

    @property
    def covid_local(self):
        return self.__covid_local

    def load_vaccine(
        self,
        df,
        date_col="date",
        labels_orig=[
            "total_vaccinations_per_hundred",
            "people_vaccinated_per_hundred",
            "people_fully_vaccinated_per_hundred"
        ]
    ):
        if labels_orig:
            df = df[[date_col, *labels_orig]]
        df = df.copy()
        df.loc[:, date_col] = pd.to_datetime(df[date_col])
        rename_cols = {
            date_col: DataCol.DATE,
            **dict(zip(labels_orig, [
                DataCol.VAC_TOTAL,
                DataCol.VAC_PEOPLE,
                DataCol.VAC_FULL
            ]))
        }
        df.rename(columns=rename_cols, inplace=True)
        self.__vaccine = df
        return self.__vaccine

    @property
    def vaccine(self):
        return self.__vaccine

    def load_test(
        self,
        df,
        date_col="Date",
        label_orig="Cumulative total"
    ):
        if label_orig:
            df = df[[date_col, label_orig]]
        df = df.copy()
        df.loc[:, date_col] = pd.to_datetime(df[date_col])
        rename_cols = {
            date_col: DataCol.DATE,
            label_orig: DataCol.TEST
        }
        df.rename(columns=rename_cols, inplace=True)
        self.__test = df
        return self.__test

    @property
    def test(self):
        return self.__test

    def load_covid_global(
        self,
        df,
        date_col="date",
        label_orig="total_cases"
    ):
        if label_orig:
            df = df[[date_col, label_orig]]
        df = df.copy()
        df.loc[:, date_col] = pd.to_datetime(df[date_col])
        rename_cols = {
            date_col: DataCol.DATE,
            label_orig: DataCol.I_TOT
        }
        df.rename(columns=rename_cols, inplace=True)
        self.__covid_global = df
        return self.__covid_global

    @property
    def covid_global(self):
        return self.__covid_global

    def load_dates(
        self,
        df,
        name=None,
        name_col="name",
        start_col="start", end_col="end",
        val_col="value"
    ):
        if name is None and DataCol.NAME not in df.columns and name_col not in df.columns:
            raise Exception("Provide name argument if dataframe doesn't have name column")
        if name is not None and (DataCol.NAME in df.columns or name_col in df.columns):
            raise Exception("Dataframe already has name column but name argument was given")
        df = DataUtil.prepare_dates(
            df,
            name_col=name_col,
            start_col=start_col,
            end_col=end_col,
            val_col=val_col
        )
        if name is not None and DataCol.NAME not in df.columns:
            df[DataCol.NAME] = pd.Series(np.array(len(df) * [name]), dtype=str)
        if DataCol.KABKO not in df.columns:
            df = df[[
                DataCol.NAME,
                DataCol.START,
                DataCol.END,
                DataCol.VAL
            ]]
            self.__dates_global = pd.concat([self.__dates_global, df])
            self.__date_names_global = self.__dates_global[DataCol.NAME].unique()
        else:
            df = df[[
                DataCol.KABKO,
                DataCol.NAME,
                DataCol.START,
                DataCol.END,
                DataCol.VAL
            ]]
            self.__dates_local = pd.concat([self.__dates_local, df])
            self.__date_names_local= self.__dates_local[DataCol.NAME].unique()
        self.__date_names = np.unique(self.__date_names_global + self.__date_names_local)
        return df

    @property
    def dates_global(self):
        return self.__dates_global

    @property
    def dates_local(self):
        return self.__dates_local

    @property
    def date_names_global(self):
        return self.__date_names_global

    @property
    def date_names_local(self):
        return self.__date_names_local

    @property
    def date_names(self):
        return self.__date_names


    def get_covid_kabko(
        self,
        kabko
    ):
        covid = self.covid_local.loc[
            self.covid_local[DataCol.KABKO] == kabko,
            [
                DataCol.DATE,
                DataCol.I,
                DataCol.R,
                DataCol.D
            ]
        ].copy()
        # del covid["kabko"]
        covid.set_index(DataCol.DATE, inplace=True)
        return covid

    def get_dates_kabko(
        self,
        kabko
    ):
        dates = pd.concat([
            self.dates_global.copy(),
            self.dates_local[self.dates_local[DataCol.KABKO] == kabko]
        ])
        return dates

    def load_population(
        self,
        df,
        kabko_col="kabko",
        label_orig="semua"
    ):
        if label_orig:
            df = df[[kabko_col, label_orig]]
        df = df.copy()
        rename_cols = {
            kabko_col: DataCol.KABKO,
            label_orig: DataCol.N
        }
        df.rename(columns=rename_cols, inplace=True)
        self.__population = df
        return self.__population

    @property
    def population(self):
        return self.__population
    
    def get_population_kabko(
        self,
        kabko
    ):
        return self.population[self.population[DataCol.KABKO] == kabko][DataCol.N].values[0]

    # Full of defaults
    # For custom, DIY
    def load_excel(
        self,
        path,
    ):
        self.load_covid_global(pd.read_excel(path, sheet_name="covid_indo"))
        self.load_covid_local(pd.read_excel(path, sheet_name="covid_jatim"))
        self.load_vaccine(pd.read_excel(path, sheet_name="vaccine"))
        self.load_test(pd.read_excel(path, sheet_name="test"))
        self.load_population(pd.read_excel(path, sheet_name="population"))
        self.load_dates(pd.read_excel(path, sheet_name="psbb"), name="psbb")
        self.load_dates(pd.read_excel(path, sheet_name="ppkm"), name="ppkm")
        self.load_dates(pd.read_excel(path, sheet_name="ppkm_mikro"), name="ppkm_mikro")
        self.load_dates(pd.read_excel(path, sheet_name="long_holiday"))
        self.load_dates(pd.read_excel(path, sheet_name="pilkada"), name="pilkada")
        self.load_dates(pd.read_excel(path, sheet_name="other_dates"))
