import numpy as np

def simulate_shutdown_restart(data, steps=30):

    # ================================
    # INPUTS
    # ================================
    T_inlet = data["T_inlet"]
    T_sea = data["T_seawater"]
    shutdown_time = data["shutdown_time"]
    P = data["P_inlet"]
    gas_gravity = data["gas_gravity"]

    # ================================
    # TIME GRID
    # ================================
    time_shutdown = np.linspace(0, shutdown_time, steps)

    # ================================
    # COOLING MODEL (SHUTDOWN)
    # ================================
    tau_cooling = 8  # thermal decay constant

    temp_shutdown = T_sea + (T_inlet - T_sea) * np.exp(-time_shutdown / tau_cooling)

    # ================================
    # RESTART HEATING
    # ================================
    time_restart = np.linspace(0, shutdown_time * 0.5, steps)

    tau_heating = 5

    temp_restart = T_inlet - (T_inlet - temp_shutdown[-1]) * np.exp(-time_restart / tau_heating)

    # ================================
    # COMBINE PROFILES
    # ================================
    full_time = np.concatenate([time_shutdown, shutdown_time + time_restart])
    full_temp = np.concatenate([temp_shutdown, temp_restart])

    # ================================
    # HYDRATE EQUILIBRIUM (PHYSICS)
    # ================================
    T_eq = 10 + 15 * np.log(P) - 25 * gas_gravity

    # ================================
    # SUBCOOLING & RISK
    # ================================
    subcooling = T_eq - full_temp
    subcooling[subcooling < 0] = 0

    # Growth proxy (time-dependent hydrate risk)
    growth = subcooling * (np.arange(len(full_temp)) + 1)

    # ================================
    # OUTPUT
    # ================================
    return {
        "time": full_time.tolist(),
        "temperature_profile": full_temp.tolist(),
        "hydrate_equilibrium": float(T_eq),
        "subcooling_profile": subcooling.tolist(),
        "hydrate_risk_profile": growth.tolist(),
        "min_temperature": float(np.min(full_temp)),
        "max_subcooling": float(np.max(subcooling))
    }