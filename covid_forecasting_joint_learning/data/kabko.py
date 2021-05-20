from . import util as DataUtil


class Kabko:
    def __init__(self, name, loader=None):
        self.name = name
        self.loader = loader

    def get_covid(self):
        covid = self.loader.covid.loc[self.loader.covid.kabko == self.name, ["date", "i", "r", "d"]].copy()
        # del covid["kabko"]
        covid = covid.set_index("date")
        return covid

    @property
    def covid(self):
        return self.get_covid()

    def get_psbb(self):
        return self.loader.psbb.loc[self.loader.psbb.kabko==self.name, ["start", "end", "value"]]

    @property
    def psbb(self):
        return self.get_psbb()

    def get_full_raw(self):
        covid = self.get_covid()
        DataUtil.add_special_dates(covid, self.get_psbb(), "psbb")

        return covid
