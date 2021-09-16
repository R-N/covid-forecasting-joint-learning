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
DELTA_IRD = [DELTA_I, DELTA_R, DELTA_D]
DELTA_SIRD = [DELTA_S, *DELTA_IRD]
I_TOT_GLOBAL = "i_tot_global"
N_GLOBAL = "n_global"
S_GLOBAL_PEOPLE = "s_global_people"
S_GLOBAL_FULL = "s_global_full"
DELTA_VAC_PEOPLE = "delta_vac_people"
DELTA_VAC_FULL = "delta_vac_full"
VAC_PEOPLE_S = "vac_people_s"
VAC_FULL_S = "vac_full_s"
DELTA_I_TOT_GLOBAL = "delta_i_tot_global"
POS_RATE = "pos_rate"
DAILY_POS_RATE = "daily_pos_rate"
BETA = "beta"
GAMMA = "gamma"
DELTA = "delta"
SIRD_VARS = [BETA, GAMMA, DELTA]
DAYS = [f"day_{i}" for i in range(7)]
DAY_DUM = "day_dum"
DAY = "day"

LOCKDOWN_DATES = [
    "psbb",
    "ppkm",
    "ppkm_mikro"
]
HOLIDAY_DATES = [
    "kenaikan_isa",
    "libur_awal_puasa",
    "hari_buruh",
    "idul_fitri_siswa",
    "idul_fitri_umum",
    "libur_semester_genap",
    "idul_adha",
    "hut_ri",
    "tahun_baru_hijriyah",
    "tahun_baru_hijriyah_ext",
    "maulid_nabi",
    "maulid_nabi_ext",
    "natal",
    "natal_ext",
    "libur_semester_ganjil",
    "tahun_baru_masehi",
    "imlek"
]
OTHER_DATES = [
    "ramadhan",
    "pilkada"
]
SINGLE_DATES = LOCKDOWN_DATES + HOLIDAY_DATES + OTHER_DATES
LABELED_DATES = {d: [d] for d in SINGLE_DATES}

DATES = list(LABELED_DATES.keys())

COLS_NON_DATE = [
    DAILY_POS_RATE,
    DELTA_VAC_PEOPLE,
    DELTA_VAC_FULL,
    VAC_PEOPLE_S,
    VAC_FULL_S,
    POS_RATE,
    DELTA_TEST
]

FUTURE_EXO_COLS = [
    DAY,
    *DAYS,
    *DATES
]
PAST_COLS = [
    *COLS_NON_DATE,
    *[c for c in FUTURE_EXO_COLS if c not in COLS_NON_DATE]
]
LABEL_COLS = [
    *SIRD_VARS
]
FINAL_SEED_COLS = [
    *SIRD
]
FINAL_COLS = [
    *IRD
]
COLS = PAST_COLS + [c for c in LABEL_COLS if c not in PAST_COLS] + [c for c in FINAL_SEED_COLS if c not in PAST_COLS]