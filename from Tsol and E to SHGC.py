import os
import pywincalc
import pandas as pd
import numpy as np


def create_flat_wavelength_data(t_sol):

    r_sol = 0.09
    wavelengths = [0.300, 2.500]
    pywincalc_wavelength_measured_data = []

    for w in wavelengths:
        direct_component = pywincalc.OpticalMeasurementComponent(
            transmittance_front=t_sol,
            transmittance_back=t_sol,
            reflectance_front=r_sol,
            reflectance_back=r_sol
        )
        pywincalc_wavelength_measured_data.append(pywincalc.WavelengthData(w, direct_component))

    return pywincalc_wavelength_measured_data


glass_material_type = pywincalc.MaterialType.MONOLITHIC
glass_material_thickness = 0.006
glass_conductivity = 1.0
glass_coated_side = pywincalc.CoatedSide.FRONT
flipped = False

shgc_environments = pywincalc.nfrc_shgc_environments()


tsol_values = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]
e_front_values = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]
e_back_values = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]

results_data = []

print(f"Starting computation... A total of {len(tsol_values) * len(e_front_values) * len(e_back_values)} operating conditions need to be simulated.")


for t_sol in tsol_values:
    for e_front in e_front_values:
        for e_back in e_back_values:

            wavelength_measurements = create_flat_wavelength_data(t_sol)
            glass_n_band_optical_data = pywincalc.ProductDataOpticalNBand(
                material_type=glass_material_type,
                thickness_meters=glass_material_thickness,
                wavelength_data=wavelength_measurements,
                coated_side=glass_coated_side,
                ir_transmittance_front=0,
                ir_transmittance_back=0,
                emissivity_front=e_front,
                emissivity_back=e_back,
                flipped=flipped
            )


            glass_thermal = pywincalc.ProductDataThermal(
                conductivity=glass_conductivity,
                thickness_meters=glass_material_thickness,
                flipped=flipped,
                opening_top=0,
                opening_bottom=0,
                opening_left=0,
                opening_right=0,
                opening_front=0
            )


            glass_layer = pywincalc.ProductDataOpticalAndThermal(glass_n_band_optical_data, glass_thermal)


            glazing_system_u = pywincalc.GlazingSystem(solid_layers=[glass_layer])
            u_value = glazing_system_u.u()

            glazing_system_shgc = pywincalc.GlazingSystem(solid_layers=[glass_layer], environment=shgc_environments)
            shgc_value = glazing_system_shgc.shgc()


            results_data.append({
                "Tsol": t_sol,
                "E1": e_front,
                "E2": e_back,
                "U_value [W/m2-K]": round(u_value, 4),
                "SHGC": round(shgc_value, 4)
            })

df = pd.DataFrame(results_data)
output_filename = "tsol+E+U+SHGC.xlsx"
df.to_excel(output_filename, index=False)

print("-" * 50)
print(f"Done.")
