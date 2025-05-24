import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift


def process_image_fourier(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Błąd: Nie można wczytać obrazu z {image_path}")
        return

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_float = img_gray.astype(np.float32)

    rows, cols = img_float.shape

    f_transform = fft2(img_float)
    f_transform_shifted = fftshift(f_transform)

    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1e-10)

    radius_low = min(rows, cols) // 10
    radius_high = min(rows, cols) // 4

    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[-center_row:rows -
                    center_row, -center_col:cols - center_col]

    mask_low = (x**2 + y**2 <= radius_low**2).astype(np.float32)

    mask_high = (x**2 + y**2 <= radius_high**2).astype(np.float32)

    f_low_freq_shifted = f_transform_shifted * mask_low
    f_high_freq_shifted = f_transform_shifted * mask_high

    img_reconstructed_low_freq = np.abs(ifft2(ifftshift(f_low_freq_shifted)))
    img_reconstructed_high_freq = np.abs(ifft2(ifftshift(f_high_freq_shifted)))

    img_reconstructed_low_freq = cv2.normalize(
        img_reconstructed_low_freq, None, 0, 255, cv2.NORM_MINMAX)
    img_reconstructed_low_freq = img_reconstructed_low_freq.astype(np.uint8)

    img_reconstructed_high_freq = cv2.normalize(
        img_reconstructed_high_freq, None, 0, 255, cv2.NORM_MINMAX)
    img_reconstructed_high_freq = img_reconstructed_high_freq.astype(np.uint8)

    profile_row_index = rows // 2

    profile_original = img_float[profile_row_index, :]
    profile_low_freq = img_reconstructed_low_freq[profile_row_index, :]
    profile_high_freq = img_reconstructed_high_freq[profile_row_index, :]

    plt.figure(figsize=(15, 12))

    plt.subplot(3, 3, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Oryginalny obraz')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Widmo Amplitud (FFT)')
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.imshow(mask_low, cmap='gray')
    plt.title(f'Maska Niskich Częstotliwości (R={radius_low})')
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.imshow(img_reconstructed_low_freq, cmap='gray')
    plt.title('Rekonstrukcja (Niskie Freq.)')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(mask_high, cmap='gray')
    plt.title(f'Maska Szerszych Częstotliwości (R={radius_high})')
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(img_reconstructed_high_freq, cmap='gray')
    plt.title('Rekonstrukcja (Szersze Freq.)')
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.plot(profile_original, label='Oryginalny Profil', color='blue')
    plt.plot(profile_low_freq,
             label=f'Profil Rekonstrukcji (Niskie Freq. R={radius_low})', color='red', linestyle='--')
    plt.plot(profile_high_freq,
             label=f'Profil Rekonstrukcji (Szersze Freq. R={radius_high})', color='green', linestyle=':')
    plt.title('Profile Liniowe wzdłuż Osi Poziomej')
    plt.ylabel('Intensywność piksela')
    plt.xlabel('Pozycja piksela')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


process_image_fourier('data/vert_lines.png')
