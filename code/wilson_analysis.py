import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import pandas as pd
from astropy.io import fits
import lightkurve as lk

import warnings

warnings.filterwarnings("ignore", category=lk.utils.LightkurveDeprecationWarning)


def design_matrix(time_vector, period_global):
    """Make a sine wave model based on a design matrix"""
    sine_term = np.sin(2.0 * np.pi * time_vector / period_global)
    cos_term = np.cos(2.0 * np.pi * time_vector / period_global)
    constant = np.ones_like(time_vector)
    linear_term = time_vector
    quadratic_term = time_vector**2
    design_matrix = np.vstack(
        [sine_term, cos_term, constant, linear_term, quadratic_term]
    ).T
    return design_matrix


def best_fit_coeffs(time, y, yerr, period_global):
    """Get the best fit coefficients from linear algebra"""
    A = design_matrix(time, period_global)
    ATA = np.dot(A.T, A / yerr[:, None] ** 2)
    mean_w = np.linalg.solve(ATA, np.dot(A.T, y / yerr**2))
    return mean_w


times = np.array(
    [
        54945.74206,
        54945.8670833,
        54946.0182163,
        54946.2335259,
        54946.3377372,
        54946.5495655,
        54946.7334679,
        54947.1662509,
        55062.8253608,
        55091.0055606,
        55123.0864583,
        55153.9797114,
        55182.0366329,
        55215.9548927,
        55216.0352649,
        55245.7660019,
        55274.7398619,
        55307.5350333,
        55336.4281441,
        55370.695297,
        55399.0571196,
        55430.8109392,
        55461.8291274,
        55492.8064711,
        55522.7621353,
        55552.0843583,
        55585.5760016,
        55614.7389024,
        55677.4444762,
        55706.6440223,
        55738.4591443,
        55769.477399,
        55801.7624176,
        55832.8010616,
        55864.8001316,
        55895.757043,
        55930.8619526,
        55958.4268791,
        55986.5230115,
        56014.5579184,
        56047.5173675,
        56077.4525238,
        56105.5895974,
        56137.5273556,
        56168.8315615,
        56203.8547584,
        56236.834641,
        56267.9141396,
        56303.6729488,
        56330.563549,
        56357.495041,
        56390.4952381,
    ]
)

ffidata = pd.read_csv("../data/f3/lc_data_new.out")

times_bkjd = times + 2400000.5 - 2454833.0

# df_sample = pd.read_csv("../data/samples/target_sample.csv", index_col=0)
# kepids = df_sample.kepid.values

# Set the KIC ID here
kepid = 5818116


# Step 1: Get the F3 lightcurve
flux = ffidata[ffidata["KIC"] == kepid].iloc[:, 1:53]
yerr = ffidata[ffidata["KIC"] == kepid].iloc[:, -52:]

result_df = pd.DataFrame(
    data={
        "time_bkjd": times_bkjd,
        "flux": flux.values.reshape(-1),
        "flux_unc": yerr.values.reshape(-1),
    }
)
result_df["lc_amp_pm30"] = np.NaN

# Step 2a: Get all the long-cadence lightcurves
sr = lk.search_lightcurve("KIC {}".format(kepid))
n_quarters = len(sr)

lcs_raw = sr.download_all()
lcs = lk.LightCurveCollection([lc.PDCSAP_FLUX.normalize() for lc in lcs_raw])

# Step 2b: Get the global period
lc_global = lcs.stitch().remove_nans().normalize()
pg = lc_global.to_periodogram()
period_global = pg.period_at_max_power.value

# Step 3: Search each each f3 time point for available long cadence
dt = 30

for i, time in enumerate(times_bkjd):
    t_lo, t_hi = time - dt, time + dt
    mask = (lc_global.time.value > t_lo) & (lc_global.time.value < t_hi)
    if mask.sum() > 13:
        lc = lc_global[mask]
        coeffs = best_fit_coeffs(
            lc.time.value, lc.flux.value, lc.flux_err.value, period_global
        )
        amplitude = np.hypot(coeffs[0], coeffs[1])
        result_df.loc[i, "lc_amp_pm30"] = amplitude


# Step 4: Save the output to disk
result_df["kepid"] = kepid

result_df.to_csv("../data/results/wilson_output.csv", index=False)
