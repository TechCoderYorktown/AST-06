import pandas as pd
import numpy as np
from scipy.stats import iqr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from scipy.stats import norm, uniform

csv_path = "M32_Final.csv"

PA_func = None
i_func = None
vrot_func = None


def load_interpolation_functions():
    global PA_func, i_func, vrot_func

    # path -> you can change this to your actual file path
    file_path = "HI.csv"
    df = pd.read_csv(file_path)

    x = df['Radius_kpc'].values
    PA_y = df['P.A._adopted'].values
    i_y = df['i_adopted'].values
    v_y = df['v_rot'].values if 'v_rot' in df.columns else None

    # linear interpolation functions
    PA_func = interp1d(x, PA_y, kind='linear', fill_value="extrapolate")
    i_func = interp1d(x, i_y, kind='linear', fill_value="extrapolate")

    if v_y is not None:
        vrot_func = interp1d(x, v_y, kind='linear', fill_value="extrapolate")
    else:
        print("⚠️ 'v_rot' does not exist in the dataset. vrot_func will not be available.")

# interpolation functions
def PA_adopted(radius_kpc):
    if PA_func is None:
        load_interpolation_functions()
    return PA_func(radius_kpc)

def i_adopted(radius_kpc):
    if i_func is None:
        load_interpolation_functions()
    return i_func(radius_kpc)

def rotational_velocity(radius_kpc):
    if vrot_func is None:
        load_interpolation_functions()
    if vrot_func is not None:
        return vrot_func(radius_kpc)
    else:
        raise ValueError("No reset here")
    

import numpy as np
import pandas as pd

def compute_projected_radius(
    file_path,
    ra_col='RA',
    dec_col='DEC',
    v_col='v_helio',
    slit_col='SLIT',
    output=True,
    save=True,
    D_M31=776.2,
    inc_deg=77.0,
    PA_deg=38.0,
    ra0_hms=(0, 42, 44.3),
    dec0_dms=(41, 16, 9)
):


    # Load CSV
    df = pd.read_csv(file_path)

    # Basic validity mask
    mask = (df[ra_col].notna() & df[dec_col].notna() &
            df[v_col].notna() & df[slit_col].notna())

    # Convert degrees
    ra_deg = df[ra_col].to_numpy()
    dec_deg = df[dec_col].to_numpy()
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

    # M31 reference center (in radians)
    ra0_deg = (ra0_hms[0] + ra0_hms[1]/60 + ra0_hms[2]/3600) * 15
    dec0_deg = dec0_dms[0] + dec0_dms[1]/60 + dec0_dms[2]/3600
    ra0_rad = np.deg2rad(ra0_deg)
    dec0_rad = np.deg2rad(dec0_deg)

    # Gnomonic projection
    delta_ra = ra_rad - ra0_rad
    denom = (np.sin(dec0_rad) * np.sin(dec_rad) +
             np.cos(dec0_rad) * np.cos(dec_rad) * np.cos(delta_ra))
    xi = (np.cos(dec_rad) * np.sin(delta_ra)) / denom
    eta = (np.cos(dec0_rad) * np.sin(dec_rad) -
           np.sin(dec0_rad) * np.cos(dec_rad) * np.cos(delta_ra)) / denom
    xi_kpc = xi * D_M31
    eta_kpc = eta * D_M31

    # Inclination and PA
    i_rad = np.deg2rad(inc_deg)
    PA_rad = np.deg2rad(PA_deg)
    alpha_j = eta_kpc * np.cos(PA_rad) + xi_kpc * np.sin(PA_rad)
    beta_j = xi_kpc * np.cos(PA_rad) - eta_kpc * np.sin(PA_rad)

    # Disk radius
    R_disk_j_0 = np.sqrt(alpha_j**2 + (beta_j / np.cos(i_rad))**2)

    # Store in DataFrame
    df['xi_kpc'] = xi_kpc
    df['eta_kpc'] = eta_kpc
    df['R_disk_kpc_init'] = np.nan
    df.loc[mask, 'R_disk_kpc_init'] = R_disk_j_0[mask]

    if output:
        print(f"[INFO] xi/eta and R_disk initialized.")
        print(f"R_disk_kpc_init: min = {df['R_disk_kpc_init'].min():.2f}, max = {df['R_disk_kpc_init'].max():.2f}")
        print(df[['xi_kpc', 'eta_kpc', 'R_disk_kpc_init']].head())

    if save:
        df.to_csv(file_path, index=False)
        print(f"[INFO] Saved updated file to: {file_path}")

    return df



def iterate_R_disk_convergence(df, max_iter=100, tol_inclination=0.01, tol_PA=0.35, D_M31=776.2, print = True):
    """
    READ CAREFULLY!!! 
    interpolate i(R), PA(R) -> convergence → R, V_rot_model_los
    """

    # initial parameters
    R_old = df['R_disk_kpc_init'].values.copy()
    xi_kpc = df['xi_kpc'].values
    eta_kpc = df['eta_kpc'].values

    for it in range(max_iter):                 
        # i_j, PA_j calculation
        i_j_rad = np.deg2rad(i_adopted(R_old))
        PA_j_rad = np.deg2rad(PA_adopted(R_old))

        # α_j, β_j calculation
        alpha_j = eta_kpc * np.cos(PA_j_rad) + xi_kpc * np.sin(PA_j_rad)
        beta_j = xi_kpc * np.cos(PA_j_rad) - eta_kpc * np.sin(PA_j_rad)

        # R_disk re-calculation
        R_new = np.sqrt(alpha_j**2 + (beta_j / np.cos(i_j_rad))**2)

        # Recalculated i, PA for new R
        i_j_new_rad = np.deg2rad(i_adopted(R_new))
        PA_j_new_rad = np.deg2rad(PA_adopted(R_new))

        # Convergence check
        inclination_change = np.abs(np.rad2deg(i_j_new_rad) - np.rad2deg(i_j_rad))
        PA_change = np.abs(np.rad2deg(PA_j_new_rad) - np.rad2deg(PA_j_rad))
        if (inclination_change < tol_inclination).all() and (PA_change < tol_PA).all():
            if print:
                print(f"Converged after {it+1} iterations.")
            break

        # Reset
        R_old = R_new.copy()

    # final θ_j calculation
    theta_j = np.arctan2(beta_j, alpha_j * np.cos(i_j_rad))  

    v_rot_M31_model = rotational_velocity(R_new)
    df['v_rot_M31_model'] = v_rot_M31_model
    df['theta_j'] = (theta_j)
    df['R_disk_kpc_final'] = R_new
    df['i_j_rad'] = (i_j_rad)
    df['PA_rad'] = (PA_j_rad)

    return df

def calculate_voffset(f_rot, v_rot_M31_model, theta_j, i_j_rad, v_helio, v_sys=-300):
    """
    Calculate velocity offset (v_helio - modeled velocity) for optimization.

    Parameters:
        f_rot (float): Scaling factor for rotation velocity
        v_rot_model (np.ndarray): Intrinsic rotation velocity model [N,]
        theta_j (np.ndarray): Azimuthal angles [radians] [N,]
        i_j_rad (float): Inclination angle in radians
        v_helio (np.ndarray): Observed heliocentric velocities [N,]
        v_sys (float): Systemic velocity [default = -300]

    Returns:
        np.ndarray: Velocity offset (v_helio - v_model_los)
    """

    # Line-of-sight projected model velocity
    v_model_los = v_sys + f_rot * v_rot_M31_model * np.cos(theta_j) * np.sin(i_j_rad)

    # Velocity offset
    v_offset = v_model_los - v_helio

    return v_offset, v_model_los

def sample_choose(df_raw, ser = False, clear = False, contam = False, target = False):
    """
    ser == True: serendipitous stars (680)
    clear == True: clear stars (1583)
    """
    if ser:
        df = df_raw[df_raw['TYPE'] == 'serendip']
    elif clear:
        df = df_raw[df_raw['CONTAM'] < 0.2]
    elif contam:
        df = df_raw[df_raw['CONTAM'] >= 0.2]
    elif target:
        df = df_raw[df_raw['TYPE'] == 'target']
    else:
        df = df_raw.copy()
    
    return df
def input_data(df):
    """
    df: DataFrame containing the data
    Returns: theta_j, i_j_rad, v_helio, v_rot_model
    """
    theta_j = df['theta_j'].values
    i_j_rad = df['i_j_rad'].values
    v_helio = df['v_helio'].values
    v_rot_M31_model = df['v_rot_M31_model'].values

    return theta_j, i_j_rad, v_helio, v_rot_M31_model













def create_prior_distributions_5D():
    priors = {}
    priors['sigma_disk'] = norm(loc=68.0, scale=26.6)  # (10, 120)

        # M31 Halo
    priors['f1'] = norm(loc=0.45, scale=0.05)
    priors['mu1'] = norm(loc=-337, scale=9.3) 
    priors['sigma1'] = norm(loc=126.8, scale=9.3) 


        # M32
    priors['f2'] = uniform(loc=0.0, scale=1)
    priors['mu2'] = norm(loc=-197.8, scale=4.4) 
    priors['sigma2'] = norm(loc=28.2, scale=4.4) 

        # MW foreground
    priors['f3'] = uniform(loc=0.0, scale=1)
    priors['mu3'] = norm(loc=-55.9, scale=8.8) 
    priors['sigma3'] = norm(loc=44.2, scale=8.8) 

        #GSS
    priors['f4'] = uniform(loc=0.0, scale=1)
    priors['mu4'] = norm(loc=-618, scale=8.8) 
    priors['sigma4'] = norm(loc=30, scale=8.8) 
        # f_rot
    priors['f_rot'] = uniform(loc=0.0, scale=1.5) 
    return priors


def input_data(df):
    """
    df: DataFrame containing the data
    Returns: theta_j, i_j_rad, v_helio, v_rot_model
    """
    theta_j = df['theta_j'].values
    i_j_rad = df['i_j_rad'].values
    v_helio = df['v_helio'].values
    v_rot_M31_model = df['v_rot_M31_model'].values

    return theta_j, i_j_rad, v_helio, v_rot_M31_model



