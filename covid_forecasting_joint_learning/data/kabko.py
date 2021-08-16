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
        datasets=None,
        datasets_torch=None,
        dataloaders=None,
        model=None
    ):
        self.__name = name
        self.__data_center = data_center
        self.__covid = covid if covid is not None else self.data_center.get_covid_kabko(self.name)
        self.__dates = dates if covid is not None else self.data_center.get_dates_kabko(self.name)
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
        self.datasets_torch = datasets_torch
        self.dataloaders = dataloaders
        self.model = model

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

    @property
    def population_global(self):
        return self.data_center.population_global

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
            datasets=self.datasets,
            datasets_torch=self.datasets_torch,
            dataloaders=self.dataloaders,
            model=self.model
        )

    @property
    def vaccine(self):
        return DataUtil.right_slice(self.data_center.vaccine, self.covid)

    @property
    def test(self):
        return DataUtil.right_slice(self.data_center.test, self.covid)

    @property
    def covid_global(self):
        return DataUtil.right_slice(self.data_center.covid_global, self.covid)

    @property
    def data_global(self):
        return DataUtil.right_slice(self.data_center.data_global, self.covid)

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
                date = self.dates[self.dates[DataCol.NAME].isin(v)][DataCol.DATE_COLS]
            else:
                date = self.dates[self.dates[DataCol.NAME] == v][DataCol.DATE_COLS]
            DataUtil.add_dates(df, date, k)

        return df

    def get_batch_sample(self, last=False):
        if last:
            *_, sample = iter(self.dataloaders[0])
        else:
            sample = next(iter(self.dataloaders[0]))
        return sample[:-1]

    def get_model_summary(self, sample_last=False):
        return self.model.get_summary(self.get_batch_sample(last=sample_last)[:5])

    def write_model_graph(self, path, sample_last=False):
        return self.model.write_graph(path, self.get_batch_sample(last=sample_last)[:5])

    def get_input_weight(self, *args, sample_last=False, **kwargs):
        return self.model.get_input_weight(self.get_batch_sample(last=sample_last)[:5], *args, **kwargs)

    def get_layer_weight(self, layer, *args, sample_last=False, **kwargs):
        return self.model.get_layer_weight(layer, self.get_batch_sample(last=sample_last)[:5], *args, **kwargs)
