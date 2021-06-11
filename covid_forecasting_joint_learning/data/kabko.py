from . import util as DataUtil
from . import cols as DataCol


class KabkoData:
    def __init__(
        self,
        name,
        data_center,
        population=None,
        covid=None,
        dates=None,
        raw=None,
        data=None,
        parent=None,
        split_indices=None,
        scaler=None,
        group=None,
        cluster=None,
        scaler_2=None,
        datasets=None
    ):
        self.__name = name
        self.__data_center = data_center
        self.__covid = covid or self.data_center.get_covid_kabko(self.name)
        self.__dates = dates or self.data_center.get_dates_kabko(self.name)
        self.__date_names = self.dates[DataCol.NAME].unique()
        self.__population = population or self.data_center.get_population_kabko(self.name)
        self.raw = raw
        self.data = data
        self.parent = parent
        self.split_indices = split_indices
        self.scaler = scaler
        self.group = group
        self.cluster = cluster
        self.scaler_2 = scaler_2
        self.datasets = datasets

    @property
    def name(self):
        return self.__name

    @property
    def data_center(self):
        return self.__data_center

    @property
    def covid(self):
        return self.__covid

    @property
    def dates(self):
        return self.__dates

    @property
    def date_names(self):
        return self.__date_names

    @property
    def population(self):
        return self.__population

    def copy(self):
        return KabkoData(
            name=self.name,
            data_center=self.data_center,
            covid=self.covid,
            dates=self.dates,
            population=self.population,
            raw=self.raw,
            data=self.data,
            parent=self,
            split_indices=self.split_indices,
            scaler=self.scaler,
            group=self.group,
            cluster=self.cluster,
            scaler_2=self.scaler_2,
            datasets=self.datasets
        )

    @property
    def vaccine(self):
        return self.data_center.vaccine

    @property
    def test(self):
        return self.data_center.test

    @property
    def covid_global(self):
        return self.data_center.covid_global

    def add_dates(
        self, df,
        dates={
            "psbb": "psbb",
            "ppkm": "ppkm",
            "ppkm_mikro": "ppkm_mikro"
        }
    ):
        for k, v in dates.items():
            if isinstance(v, list):
                date = dates[dates[DataCol.NAME].isin(v)]
            else:
                date = dates[dates[DataCol.NAME] == v]
            DataUtil.add_dates(df, date, k)

        return df
