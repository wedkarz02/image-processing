import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_profile_with_sampling(csv_file_path, sampling_frequency_pixels=50, output_filename="profile_with_sampling.png"):
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"err: '{csv_file_path}' not found")
        return
    except Exception as e:
        print(f"err: {e}")
        return

    distance_col = 'Distance_(pixels)'
    gray_value_col = 'Gray_Value'

    if distance_col not in df.columns or gray_value_col not in df.columns:
        print(
            f"err: columns '{distance_col}', '{gray_value_col}' not found")
        print(f"Dostępne kolumny: {df.columns.tolist()}")
        return

    plt.figure(figsize=(12, 7))
    plt.plot(df[distance_col], df[gray_value_col], color='black',
             linewidth=1.5, label='Oryginalny profil liniowy')

    sampled_indices = np.arange(0, len(df), sampling_frequency_pixels)
    sampled_distance = df[distance_col].iloc[sampled_indices]
    sampled_gray_value = df[gray_value_col].iloc[sampled_indices]

    plt.scatter(sampled_distance, sampled_gray_value, color='red', marker='o', s=50,
                zorder=5, label=f'Punkty próbkowania (co {sampling_frequency_pixels} px)')

    for dist, val in zip(sampled_distance, sampled_gray_value):
        plt.axvline(x=dist, color='gray', linestyle='--',
                    linewidth=0.8, alpha=0.6)

    plt.title(
        'Profil liniowy obrazu z zaznaczonymi punktami próbkowania', fontsize=16)
    plt.xlabel('Distance (pixels)', fontsize=14)
    plt.ylabel('Gray Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    try:
        plt.savefig(output_filename, dpi=300)
        print(f"Plot saved as: '{output_filename}'")
    except Exception as e:
        print(f"err: {e}")


if __name__ == "__main__":
    example_csv_file = 'data/Plot-Values.csv'
    plot_profile_with_sampling(example_csv_file, sampling_frequency_pixels=50,
                               output_filename="output/fresnel_profile_sampling.png")
