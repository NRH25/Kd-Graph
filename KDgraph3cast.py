import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

file_paths = [
    '/Users/nicholasrodchenko/Desktop/Soka/Kdcalc/2022_04_11/SagamiBay_CAST_001_2022_04_11_095037_URC_cast1.csv',
    '/Users/nicholasrodchenko/Desktop/Soka/Kdcalc/2022_04_11/SagamiBay_CAST_002_2022_04_11_095823_URC_cast2.csv',
    '/Users/nicholasrodchenko/Desktop/Soka/Kdcalc/2022_04_11/SagamiBay_CAST_003_2022_04_11_100634_URC_cast3.csv'
]

Es_0_values = {
    0: {'443': 103.9043, '490': 112.6771, '665': 93.37347},  
    1: {'443': 114.1896, '490': 124.6543, '665': 105.316},  
    2: {'443': 116.5031, '490': 127.1162, '665': 107.2043}   
}

colors = ['blue', 'green', 'red']

fig_443, ax_443 = plt.subplots(figsize=(10, 6))
fig_490, ax_490 = plt.subplots(figsize=(10, 6))
fig_665, ax_665 = plt.subplots(figsize=(10, 6))

Kd_443_values = []
Kd_490_values = []
Kd_665_values = []

for i, file_path in enumerate(file_paths):
    data = pd.read_csv(file_path, encoding='latin1')
    
    print(f"Data from cast {i+1}:")
    print(data.head())
    print(data.dtypes)
    
    pitch_column_index = data.columns.get_loc('EdZPitch')  
    roll_column_index = data.columns.get_loc('EdZRoll')  
    depth_column_index = data.columns.get_loc('LuZDepth (m)')  
    EdZ_443_column_index = data.columns.get_loc('EdZ443 ')  
    EdZ_490_column_index = data.columns.get_loc('EdZ490 ')  
    EdZ_665_column_index = data.columns.get_loc('EdZ665 ')  
    Es_443_z_column_index = data.columns.get_loc('Ed0443 ')  
    Es_490_z_column_index = data.columns.get_loc('Ed0490 ')  
    Es_665_z_column_index = data.columns.get_loc('Ed0665 ')  
    
    data.iloc[:, pitch_column_index] = pd.to_numeric(data.iloc[:, pitch_column_index], errors='coerce')
    data.iloc[:, roll_column_index] = pd.to_numeric(data.iloc[:, roll_column_index], errors='coerce')
    
    reliable_data = data[(data.iloc[:, pitch_column_index].abs() <= 5) & (data.iloc[:, roll_column_index].abs() <= 5)]

    depth = reliable_data.iloc[:, depth_column_index]
    EdZ_443 = reliable_data.iloc[:, EdZ_443_column_index]
    EdZ_490 = reliable_data.iloc[:, EdZ_490_column_index]
    EdZ_665 = reliable_data.iloc[:, EdZ_665_column_index]

    Es_443_z = reliable_data.iloc[:, Es_443_z_column_index]
    Es_490_z = reliable_data.iloc[:, Es_490_z_column_index]
    Es_665_z = reliable_data.iloc[:, Es_665_z_column_index]

    Es_443_0 = Es_0_values[i]['443']
    Es_490_0 = Es_0_values[i]['490']
    Es_665_0 = Es_0_values[i]['665']

    EdZ_443_corrected = EdZ_443 * (Es_443_0 / Es_443_z)
    EdZ_490_corrected = EdZ_490 * (Es_490_0 / Es_490_z)
    EdZ_665_corrected = EdZ_665 * (Es_665_0 / Es_665_z)

    valid_443 = (EdZ_443_corrected > 0) & (depth >= 0) & (depth <= 20)
    valid_490 = (EdZ_490_corrected > 0) & (depth >= 0) & (depth <= 20)
    valid_665 = (EdZ_665_corrected > 0) & (depth >= 0) & (depth <= 10)

    depth_443 = depth[valid_443]
    depth_490 = depth[valid_490]
    depth_665 = depth[valid_665]

    ln_EdZ_443 = np.log(EdZ_443_corrected[valid_443])
    ln_EdZ_490 = np.log(EdZ_490_corrected[valid_490])
    ln_EdZ_665 = np.log(EdZ_665_corrected[valid_665])

    slope_443, intercept_443, r_value_443, p_value_443, std_err_443 = stats.linregress(depth_443, ln_EdZ_443)
    slope_490, intercept_490, r_value_490, p_value_490, std_err_490 = stats.linregress(depth_490, ln_EdZ_490)
    slope_665, intercept_665, r_value_665, p_value_665, std_err_665 = stats.linregress(depth_665, ln_EdZ_665)

    Kd_443 = -slope_443
    Kd_490 = -slope_490
    Kd_665 = -slope_665

    Kd_443_values.append(Kd_443)
    Kd_490_values.append(Kd_490)
    Kd_665_values.append(Kd_665)

    ax_443.scatter(depth_443, ln_EdZ_443, color=colors[i], label=f'Cast {i+1} Data', s=2)
    ax_443.plot(depth_443, slope_443 * depth_443 + intercept_443, color=colors[i])
    ax_490.scatter(depth_490, ln_EdZ_490, color=colors[i], label=f'Cast {i+1} Data', s=2)
    ax_490.plot(depth_490, slope_490 * depth_490 + intercept_490, color=colors[i])
    ax_665.scatter(depth_665, ln_EdZ_665, color=colors[i], label=f'Cast {i+1} Data', s=2)
    ax_665.plot(depth_665, slope_665 * depth_665 + intercept_665, color=colors[i])

for ax, title in zip([ax_443, ax_490, ax_665], 
                     ['443 nm', '490 nm', '665 nm']):
    ax.set_xlabel('Depth (z) (m)')
    ax.set_ylabel('ln(Ed(z)) (µW/cm² nm)')
    ax.set_title(f'ln(Ed(z)) vs. Depth for {title}')
    ax.legend()
    ax.grid(True)

ax_443.set_ylim(-3, 5)
ax_490.set_ylim(-3, 5)
ax_665.set_ylim(-3, 5)

ax_443.set_xlim(0,10)
ax_490.set_xlim(0,10)
ax_665.set_xlim(0,10)

plt.show()

average_Kd_443 = np.mean(Kd_443_values)
average_Kd_490 = np.mean(Kd_490_values)
average_Kd_665 = np.mean(Kd_665_values)

print(f'The average attenuation coefficient (Kd) for 443 nm is: {average_Kd_443} m^-1')
print(f'The average attenuation coefficient (Kd) for 490 nm is: {average_Kd_490} m^-1')
print(f'The average attenuation coefficient (Kd) for 665 nm is: {average_Kd_665} m^-1')
