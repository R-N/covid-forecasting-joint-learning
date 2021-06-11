S = "s"
I = "i"
R = "r"
D = "d"
N = "n"
I_TOT = "i_tot"
VAC_TOTAL = "vac_total"
VAC_PEOPLE = "vac_people"
VAC_FULL = "vac_full"
DATE = "date"
KABKO = "kabko"
NAME = "name"
START = "start"
END = "end"
VAL = "value"
TEST = "test"
DATE_COLS = [
    START,
    END,
    VAL
]
DATES_GLOBAL = [NAME, *DATE_COLS]
DATES_LOCAL = [KABKO, *DATES_GLOBAL]
