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
        self.name = name
        self.data_center = data_center
        self.covid = covid or self.data_center.get_covid_kabko(self.name)
        self.dates = dates or self.data_center.get_dates_kabko(self.name)
        self.population = population or self.data_center.get_population_kabko(self.name)
        self.raw = raw
        self.data = data
        self.parent = parent
        self.split_indices = split_indices
        self.scaler = scaler
        self.group = group
        self.cluster = cluster
        self.scaler_2 = scaler_2
        self.datasets = datasets


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

    @property
    def date_names(self):
        return self.dates[DataCol.NAME].unique()

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
                date = dates[dates[DataCol.name].isin(v)]
            else:
                date = dates[dates[DataCol.name] == v]
            DataUtil.add_dates(df, date, k)

        return df
