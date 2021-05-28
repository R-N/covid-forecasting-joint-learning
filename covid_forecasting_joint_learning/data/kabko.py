from . import util as DataUtil


class KabkoData:
    def __init__(
        self,
        name,
        data_center=None,
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


    def get_covid(
        self,
        kabko_col="kabko",
        date_col="date",
        labels=["i", "r", "d"]
    ):
        covid = self.data_center.covid.loc[
            self.data_center.covid[kabko_col] == self.name,
            [date_col, *labels]
        ].copy()
        # del covid["kabko"]
        covid.set_index(date_col, inplace=True)
        return covid

    @property
    def covid(self):
        return self.get_covid()

    def get_psbb(
        self,
        kabko_col="kabko",
        start_col="start",
        end_col="end",
        val_col="value"
    ):
        return self.data_center.psbb.loc[
            self.data_center.psbb[kabko_col] == self.name,
            [start_col, end_col, val_col]
        ]

    @property
    def psbb(self):
        return self.get_psbb()

    def add_special_dates(self, df, psbb_col="psbb"):
        DataUtil.add_special_dates(df, self.get_psbb(), psbb_col)
        return df
