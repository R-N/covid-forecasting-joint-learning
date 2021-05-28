import requests
import pandas as pd
# from datetime import datetime
from multiprocessing.pool import ThreadPool


class DataGetter:
    def __init__(self, endpoint):
        self.endpoint = endpoint if endpoint.endswith("/") else endpoint + "/"

    def generate_date_range(self, start, end):
        dates = pd.date_range(start=start,end=end).to_pydatetime().tolist()
        dates = [d.strftime("%Y-%m-%d") for d in dates]
        return dates

    def get_data(self, date):
        resp = requests.get(self.endpoint + "%s/%s/" % (date, date))
        if resp.status_code != 200:
            # This means something went wrong.
            raise Exception("GET {}".format(resp.status_code))
        data = resp.json()
        data = [{
            "kabko": x["kabko"],
            "date": date,
            "infected_total": int(x["confirm"]),
            "recovered": int(x["confirm_selesai"]),
            "dead": int(x["meninggal_rtpcr"])
        } for x in data]
        data = [{
            **x,
            "infected": (x["infected_total"]-x["recovered"]-x["dead"])
        } for x in data]
        return data

    def get_data_bulk(
        self,
        dates,
        max_process_count=366,
        max_tasks_per_child=100,
        pool=None
    ):

        args = [(d,) for d in dates]

        if max_process_count == 1:
            return [self.get_data(*a) for a in args]

        # we prepare the pool
        # pool in this context is like collection of available tabs
        # pool = pool or Pool(processes=max_process_count, maxtasksperchild=max_tasks_per_child)
        pool = pool or ThreadPool(processes=min(len(args), max_process_count))

        # now we execute it
        # we use starmap instead of map because there are multiple arguments
        try:
            output = pool.starmap(self.get_data, args)
            pool.close()
            pool.join()
        except ConnectionError:
            raise
        finally:
            pool.terminate()
            del pool
        output = [d for data in output for d in data]
        output = sorted(output, key=lambda x: (x["kabko"], x["date"]))
        return output

    def to_dataframe(self, data):
        return pd.DataFrame.from_records(data)
