S = "s"
I = "i"
R = "r"
D = "d"
N = "n"
I_TOT = "i_tot"
VAC_PEOPLE = "vac_people"
VAC_FULL = "vac_full"
VAC_ALL = [VAC_PEOPLE, VAC_FULL]
DATE = "date"
KABKO = "kabko"
NAME = "name"
START = "start"
END = "end"
VAL = "value"
TEST = "test"
DELTA_TEST = "delta_test"
DATE_COLS = [
    START,
    END,
    VAL
]
DATES_GLOBAL = [NAME, *DATE_COLS]
DATES_LOCAL = [KABKO, *DATES_GLOBAL]
IRD = [I, R, D]
SIRD = [S, I, R, D]
SIRDN = [S, I, R, D, N]
DELTA_S = "delta_s"
DELTA_I = "delta_i"
DELTA_R = "delta_r"
DELTA_D = "delta_d"
I_TOT_GLOBAL = "i_tot_global"
N_GLOBAL = "n_global"
S_GLOBAL_PEOPLE = "s_global_people"
S_GLOBAL_FULL = "s_global_full"
DELTA_VAC_PEOPLE = "delta_vac_people"
DELTA_VAC_FULL = "delta_vac_full"
VAC_PEOPLE_S = "vac_people_s"
VAC_FULL_S = "vac_full_s"
BETA = "beta"
GAMMA = "gamma"
DELTA = "delta"
SIRD_VARS = [BETA, GAMMA, DELTA]
