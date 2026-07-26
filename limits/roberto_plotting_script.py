import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import os
import argparse
import re

hep.style.use("CMS")

def txt_to_csv(txt_file, rinv_value="0p3"):
    with open(txt_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    data = {}
    for line in lines:
        parts = line.split('\t')
        key = parts[0]
        values = [float(v) for v in parts[1:] if v]
        data[key] = values
    
    rows = []
    x_vals = data.get('mass', [])
    for i in range(len(x_vals)):
        row = {
            'x_val': x_vals[i],
            'rinv': rinv_value,
            'obs_lim': data['y_observed'][i],
            'expected_m2sigma': data['y_down_points2'][i],
            'expected_m1sigma': data['y_down_points1'][i],
            'expected': data['y_vals'][i],
            'expected_p1sigma': data['y_up_points1'][i],
            'expected_p2sigma': data['y_up_points2'][i]
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

def parse_limits_file(file_path, rinv_value="0p3"):
    _, ext = os.path.splitext(file_path)
    
    if ext == '.txt':
        return txt_to_csv(file_path, rinv_value)
    elif ext == '.csv':
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def extract_params_from_filename(filename):
    filename_no_ext = os.path.splitext(filename)[0]
    match = re.search(r'(mMed|rinv)_(.+)', filename_no_ext)
    if not match:
        raise ValueError(f"Cannot parse filename: {filename}")
    
    poi = match.group(1)
    params_str = match.group(2)
    params = {}
    
    for param in params_str.split('-'):
        param = param.strip()
        if param:
            key_match = re.match(r'([a-zA-Z]+)([\d.p]+)', param)
            if key_match:
                key, value = key_match.groups()
                params[key] = value
    
    return poi, params

def get_xsec_dict(mmed_values):
    xsec_map = {
        500: 41.76, 600: 19.65, 700: 10.38, 800: 6.006, 900: 3.717,
        1000: 2.413, 1500: 0.4832, 2000: 0.1634, 2500: 0.07178,
        3000: 0.03603, 3500: 0.01999, 4000: 0.01185,
    0.0: 0.08492,
    0.05: 0.08492,
    0.1: 0.08492,
    0.2: 0.08492,
    0.3: 0.08492,
    0.4: 0.08492,
    0.5: 0.08492,
    0.6: 0.08492,
    0.7: 0.08492,
    0.8: 0.08492,
    0.9: 0.08492,
    1.0: 0.08492,
    }
    return {m: xsec_map.get(int(m)) for m in mmed_values if int(m) in xsec_map}

def process_file(file_path, args, poi, file_params, suffix=""):
    if poi == 'mMed':
        rinv = args.rinv or file_params.get('rinv', '0p3')
        mdark = args.mdark or int(file_params.get('mDark', file_params.get('mdark', '20')))
        yukawa = args.yukawa or float(file_params.get('yukawa', '1'))
        rinv_display = rinv.replace('p', '.')
        legend_param = fr"$r_{{inv}}={rinv_display}, m_{{dark}}={mdark}\ GeV, y={yukawa}$"
    else:
        mmed = args.mmed or int(file_params.get('mMed', '1000'))
        mdark = args.mdark or int(file_params.get('mDark', file_params.get('mdark', '20')))
        yukawa = args.yukawa or float(file_params.get('yukawa', '1'))
        legend_param = fr"$m_{{med}}={mmed}\ GeV, m_{{dark}}={mdark}\ GeV, y={yukawa}$"
    
    df = parse_limits_file(file_path, rinv if poi == 'mMed' else '0p3')
    x_values = df["x_val"].values
    
    if poi == 'mMed':
        xsec = get_xsec_dict(x_values)
        x_label = r"$M_{med}$ (GeV)"
        expected = df["expected"].values * np.array([xsec.get(int(m), 1) for m in x_values])
        observed = df["obs_lim"].values * np.array([xsec.get(int(m), 1) for m in x_values])
        expected_up_1sigma = df["expected_p1sigma"].values * np.array([xsec.get(int(m), 1) for m in x_values])
        expected_down_1sigma = df["expected_m1sigma"].values * np.array([xsec.get(int(m), 1) for m in x_values])
        expected_up_2sigma = df["expected_p2sigma"].values * np.array([xsec.get(int(m), 1) for m in x_values])
        expected_down_2sigma = df["expected_m2sigma"].values * np.array([xsec.get(int(m), 1) for m in x_values])
        y_min, y_max = min(xsec.values()) * 0.8, max(xsec.values()) * 1.2
    else:
        x_label = r"$r_{inv}$"
        xsec = get_xsec_dict(x_values)
        expected = df["expected"].values
        observed = df["obs_lim"].values
        expected_up_1sigma = df["expected_p1sigma"].values
        expected_down_1sigma = df["expected_m1sigma"].values
        expected_up_2sigma = df["expected_p2sigma"].values
        expected_down_2sigma = df["expected_m2sigma"].values
        y_min, y_max = 0.002, 200

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(x_values, expected, color='black', linestyle='--', label='Expected')
    ax.plot(x_values, observed, color='black', marker='o', markersize=5, label='Observed')
    ax.fill_between(x_values, expected_up_2sigma, expected_down_2sigma, color='#85D1FBff', label='95% CL expected')
    ax.fill_between(x_values, expected_up_1sigma, expected_down_1sigma, color='#FFDF7Fff', label='68% CL expected')

    ax.plot(x_values, [xsec.get(int(m), 1) for m in x_values], color='red', linestyle='-', label='Theory')


    leg = ax.legend(title=legend_param, 
                    fontsize=18, title_fontsize=24, ncol=2, frameon=True, edgecolor='white', loc = "best")
    # leg._legend_box.align = "left"
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel(r'$\sigma\times\mathcal{BR}$ (pb)', fontsize=24)
    ax.set_xlim(min(x_values), max(x_values))
    ax.set_ylim(y_min, y_max)
    ax.set_yscale('log')
    

    hep.cms.label("Preliminary", ax=ax, data=True, lumi=args.lumi, fontsize=18)

    output_file = args.output if args.output else f"Limits_for_AN_updated_V2/limits_{poi}_{'-'.join(f'{k}{v}' for k,v in file_params.items())}.pdf"
    if suffix != "":
        output_file = output_file.replace('.pdf', f'_{suffix}.pdf')
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot limits from txt or csv file(s)')
    parser.add_argument('inFile', help='Input file (txt or csv) or folder')
    parser.add_argument('--rinv', default=None, help='Rinv value (overrides filename)')
    parser.add_argument('--mmed', type=int, default=None, help='Mediator mass in GeV')
    parser.add_argument('--mdark', type=int, default=None, help='Dark mass in GeV')
    parser.add_argument('--yukawa', type=str, default=None, help='Yukawa coupling')
    parser.add_argument('--lumi', type=float, default=138, help='Luminosity in fb^-1 (default: 138)')
    parser.add_argument('--output', default=None, help='Output plot filename (default: auto-generated)')
    parser.add_argument('--output_suffix', default='', help='Suffix to add to output filename')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.inFile):
        files = [f for f in os.listdir(args.inFile) if f.endswith(('.txt', '.csv'))]
        for file in sorted(files):
            file_path = os.path.join(args.inFile, file)
            try:
                poi, file_params = extract_params_from_filename(file)
                process_file(file_path, args, poi, file_params, suffix=args.output_suffix)
            except Exception as e:
                print(f"Error processing {file}: {e}")
    else:
        filename = os.path.basename(args.inFile)
        poi, file_params = extract_params_from_filename(filename)
        print(f"suffix - {args.output_suffix}")
        process_file(args.inFile, args, poi, file_params, suffix=args.output_suffix)


if __name__ == "__main__":
    main()