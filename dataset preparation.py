# -*- coding: utf-8 -*-
import os
import subprocess
import pandas as pd
import multiprocessing
import numpy as np
import shutil
import csv
from tqdm import tqdm
from eppy.modeleditor import IDF
from ladybug.epw import EPW
from scipy.stats import qmc
from itertools import product


EP_ROOT = r"D:\EnergyPlusV9-1-0"
EP_PATH = os.path.join(EP_ROOT, "energyplus.exe")
IDD_PATH = os.path.join(EP_ROOT, "Energy+.idd")
READ_VARS_EXE = os.path.join(EP_ROOT, "PostProcess", "ReadVarsESO.exe")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IDF_TEMPLATE = os.path.join(SCRIPT_DIR, "model.idf")
EPW_FOLDER = os.path.join(SCRIPT_DIR, "EPW files")
WORK_DIR = os.path.join(SCRIPT_DIR, "batch_runs_global")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "dataset.csv")


N_SAMPLES_PER_CITY = 30
RADIATION_THRESHOLD = 1.0
J_TO_GJ = 1_000_000_000


NUM_CORES = max(1, multiprocessing.cpu_count() - 5)
BATCH_SIZE = 50



def get_climate_info(epw_path):

    try:
        epw = EPW(epw_path)
        temp = epw.dry_bulb_temperature.values
        # HDD18 & CDD10
        hdd = sum([max(18 - t, 0) for t in temp]) / 24.0
        cdd = sum([max(t - 10, 0) for t in temp]) / 24.0
        ghi = sum(epw.global_horizontal_radiation.values) / 1000.0
        t_ave = np.mean(temp)
        is_daytime = [r > RADIATION_THRESHOLD for r in epw.global_horizontal_radiation.values]
        return {
            'latitude': round(epw.location.latitude, 2), 'hdd': round(hdd, 1), 'cdd': round(cdd, 1),
            'ghi': round(ghi, 2), 't_ave': round(t_ave, 2), 'daytime_mask': is_daytime
        }
    except:
        return None


def process_csv_energy(csv_path, mask):

    try:
        if not os.path.exists(csv_path): return None
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        mask = np.array(mask)


        c_cols = [c for c in df.columns if 'Cooling' in c and '[J]' in c]
        h_cols = [c for c in df.columns if 'Heating' in c and 'Gas' in c and '[J]' in c]
        if not h_cols:
            h_cols = [c for c in df.columns if 'NaturalGas' in c and '[J]' in c]
        if not h_cols:
            h_cols = [c for c in df.columns if 'Heating' in c and '[J]' in c]

        res = {}
        for prefix, col in [('C', c_cols[0] if c_cols else None), ('H', h_cols[0] if h_cols else None)]:
            if col:
                data = df[col].values
                res[f'{prefix}_Total_GJ'] = round(data.sum() / J_TO_GJ, 4)
                res[f'{prefix}_Day_GJ'] = round(data[mask].sum() / J_TO_GJ, 4)
                res[f'{prefix}_Night_GJ'] = round(data[~mask].sum() / J_TO_GJ, 4)
            else:
                res[f'{prefix}_Total_GJ'] = 0.0
        return res
    except:
        return None


def generate_lhs_combos(n_samples):


    n_samples = min(n_samples, 2025)
    unique_combos = set()

    lower_bounds = [0.1, 0.1, 0.1, 0.1]
    upper_bounds = [0.9, 0.9, 0.9, 0.9]


    bounds_pairs = list(zip(lower_bounds, upper_bounds))
    raw_corners = np.array(list(product(*bounds_pairs)))

    for row in raw_corners:

        t_c, t_d = max(row[0], row[1]), min(row[0], row[1])
        e_c, e_d = min(row[2], row[3]), max(row[2], row[3])

        t_c_r = round(float(np.round(t_c, 1)), 1)
        t_d_r = round(float(np.round(t_d, 1)), 1)
        e_c_r = round(float(np.round(e_c, 1)), 1)
        e_d_r = round(float(np.round(e_d, 1)), 1)


        unique_combos.add((t_c_r, e_c_r, t_d_r, e_d_r))


    if len(unique_combos) >= n_samples:
        return list(unique_combos)[:n_samples]


    sampler = qmc.LatinHypercube(d=4, seed=42)

    while len(unique_combos) < n_samples:
        needed = n_samples - len(unique_combos)

        raw_samples = sampler.random(n=needed * 3)
        scaled = qmc.scale(raw_samples, lower_bounds, upper_bounds)

        for row in scaled:
            t_c, t_d = max(row[0], row[1]), min(row[0], row[1])
            e_c, e_d = min(row[2], row[3]), max(row[2], row[3])

            t_c_r = round(float(np.round(t_c, 1)), 1)
            t_d_r = round(float(np.round(t_d, 1)), 1)
            e_c_r = round(float(np.round(e_c, 1)), 1)
            e_d_r = round(float(np.round(e_d, 1)), 1)


            if 0.1 <= t_c_r <= 0.9 and 0.1 <= t_d_r <= 0.9 and \
                    0.1 <= e_c_r <= 0.9 and 0.1 <= e_d_r <= 0.9:
                if t_c_r >= t_d_r and e_c_r <= e_d_r:
                    unique_combos.add((t_c_r, e_c_r, t_d_r, e_d_r))
                    if len(unique_combos) == n_samples:
                        break

    return list(unique_combos)



def run_simulation(args):
    idx, epw_path, t_clear, e_clear, t_dark, e_dark, climate_data, city_name, task_uid = args
    case_dir = os.path.join(WORK_DIR, f"task_{idx}")
    os.makedirs(case_dir, exist_ok=True)

    try:
        IDF.setiddname(IDD_PATH)
        idf = IDF(IDF_TEMPLATE)


        for mat_name, t_val, e_val in [("clear", t_clear, e_clear), ("dark", t_dark, e_dark)]:
            mat = idf.getobject('WINDOWMATERIAL:GLAZING', mat_name)
            if mat:
                mat.Solar_Transmittance_at_Normal_Incidence = t_val
                mat.Front_Side_Infrared_Hemispherical_Emissivity = e_val

        temp_idf = os.path.join(case_dir, "in.idf")
        idf.saveas(temp_idf)


        subprocess.run([EP_PATH, "-w", epw_path, "-d", case_dir, temp_idf],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        if os.path.exists(READ_VARS_EXE):
            subprocess.run([READ_VARS_EXE, " ", "unlimited"], cwd=case_dir, stdout=subprocess.PIPE)

        energy = process_csv_energy(os.path.join(case_dir, "eplusout.csv"), climate_data['daytime_mask'])

        if energy:
            res = {
                'task_id': task_uid, 'City': city_name,
                'T_clear': t_clear, 'E_clear': e_clear,
                'T_dark': t_dark, 'E_dark': e_dark,
                'Delta_T': round(t_clear - t_dark, 2),
                'Delta_E': round(e_dark - e_clear, 2)
            }
            res.update({k: climate_data[k] for k in ['latitude', 'hdd', 'cdd', 'ghi', 't_ave']})
            res.update(energy)
            res['Total_HVAC_GJ'] = res.get('C_Total_GJ', 0) + res.get('H_Total_GJ', 0)
            return res
    except:
        return None
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


# ================= 🚀 4. 主程序控制 =================

def main():
    if not os.path.exists(WORK_DIR): os.makedirs(WORK_DIR)

    # 1. 断点检测
    done_tasks = set()
    file_exists = os.path.exists(OUTPUT_FILE)
    if file_exists:
        try:
            temp_df = pd.read_csv(OUTPUT_FILE, usecols=['task_id'])
            done_tasks = set(temp_df['task_id'].astype(str).tolist())
            print(f"Breakpoint detected. Skipped {len(done_tasks)} tasks")
        except:
            pass



    glazing_combos = generate_lhs_combos(N_SAMPLES_PER_CITY)


    epw_files = [os.path.join(EPW_FOLDER, f) for f in os.listdir(EPW_FOLDER) if f.endswith('.epw')]

    all_tasks = []

    for epw_p in epw_files:
        city = os.path.basename(epw_p).split('__')[-1].replace('.epw', '')
        climate = get_climate_info(epw_p)
        if climate:
            for t_c, e_c, t_d, e_d in glazing_combos:
                task_uid = f"{city}_tc{t_c}_ec{e_c}_td{t_d}_ed{e_d}"
                if task_uid not in done_tasks:
                    all_tasks.append((len(all_tasks), epw_p, t_c, e_c, t_d, e_d, climate, city, task_uid))

    if not all_tasks:
        print("Complete!!")
        return

    print(f"Parallel engine started | Cores: {NUM_CORES} | Remaining simulation tasks: {len(all_tasks)}")

    temp_results = []
    with multiprocessing.Pool(processes=NUM_CORES) as pool:
        with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8-sig') as f:
            writer = None

            pbar = tqdm(pool.imap_unordered(run_simulation, all_tasks), total=len(all_tasks))
            for r in pbar:
                if r:
                    temp_results.append(r)

                if len(temp_results) >= BATCH_SIZE:
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=temp_results[0].keys())
                        if not file_exists or os.stat(OUTPUT_FILE).st_size == 0:
                            writer.writeheader()
                            file_exists = True

                    writer.writerows(temp_results)
                    f.flush()
                    temp_results = []

            if temp_results:
                if writer:
                    writer.writerows(temp_results)
                elif not file_exists:
                    writer = csv.DictWriter(f, fieldnames=temp_results[0].keys())
                    writer.writeheader()
                    writer.writerows(temp_results)
                f.flush()

    shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()