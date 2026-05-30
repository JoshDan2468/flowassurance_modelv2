import numpy as np

def compute_features(data, transient):

    # Extract inputs
    T_inlet = data["T_inlet"]
    P_inlet = data["P_inlet"]
    Q_gas = data["Q_gas"]
    Q_oil = data["Q_oil"]
    Q_water = data["Q_water"]
    D_pipe = data["D_pipe"]
    T_sea = data["T_seawater"]
    gas_gravity = data["gas_gravity"]
    chemical_injection = data["chemical_injection"]
    shutdown_time = data["shutdown_time"]

    # ================================
    # REBUILD TRAINING FEATURES
    # ================================

    # Hydrate equilibrium temp
    T_eq = 10 + 15 * np.log(P_inlet) - 25 * gas_gravity

    # Subcooling
    subcooling = T_eq - T_inlet

    # Final temperature after shutdown
    T_final = T_sea + (T_inlet - T_sea) * np.exp(-shutdown_time / 8)

    # Time below hydrate zone
    time_below_eq = max(0, shutdown_time if T_eq > T_final else 0)

    # Growth factor
    growth_factor = time_below_eq * subcooling

    # Water fraction
    water_fraction = Q_water / (Q_water + Q_oil + 1e-6)

    # Chemical inhibition
    inhibition_effect = np.exp(-chemical_injection / 50)

    # Flow calculations
    liquid_flow = Q_oil + Q_water
    gas_velocity = Q_gas / (D_pipe**2 + 1e-6)

    # ================================
    # BUILD FINAL FEATURE VECTOR
    # ================================

    features = data.copy()

    features.update({
        "T_eq": T_eq,
        "subcooling": subcooling,
        "T_final": T_final,
        "time_below_eq": time_below_eq,
        "growth_factor": growth_factor,
        "water_fraction": water_fraction,
        "inhibition_effect": inhibition_effect,
        "liquid_flow": liquid_flow,
        "gas_velocity": gas_velocity
    })

    return features