# Name: Sarvansh Mishra
# Reg. No.: 24BCE0995
'''
1. read_xrd_data(file_path) function:
This function reads the data from a txt file which contains the data from the XRD experiment, the file should contain the data such that each line contains a value for x-axis and y-axis respectively with x-axis representing the 2θ and y-axis representing Intensity values.

2. baseline_correction(x,y,degrees=2) function:
This function removes the background noise using a polynomial fit, the default polynomial fit being used here has a degree 2 representing the quadratic fitting and the baseline thus generated is subtracted from the data. The formula used is:
Baseline(x) = a2x2 + a1x + a0
Where a2, a1, and a0 are polynomial coefficients obtained from numpy.polynomial.polynomial.fit().

3. detect_peaks(x,y) function:
It identifies the peak positions in the fitted data. A Savitzky-Golay filter is used for smoothing the data obtained. It calls the find_peaks() function with prominence-based filtering to detect peaks. It returns the detected peak indices and smoothed intensity values.

4. pseudo_voigt(x,h,x0,sigma,gamma,eta) function:
It defines a pseudo Voigt function for peak fitting.

5. fit_peaks(x,y,peaks) function:
It fits the detected peaks using the pseudo voigt function. For each peak that is detected it initializes the parameters and fits it using the curve_fit() function. Finally, it returns the refined peak positions and fitting parameters.

6. scherrer_equation(fwhm, wavelength=1.5406, theta=0.0, shape_factor=0.9) function:
This function calculates the crystallite size using the Scherrer equation. 
It also calculates the FWHM approximation for gaussian component:

7. calculate_goodness_of_fit(y_obs,y_cal) function:
This function calculates the goodness of the fit for the fitted curve. The formula used to calculate the goodness of the fit is:

Thus, that there explains the entire working of the program.

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from numpy.polynomial import Polynomial

def pseudo_voigt(x, h, x0, sigma, gamma, eta):
    gaussian = np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    lorentzian = (gamma / 2) ** 2 / ((x - x0) ** 2 + (gamma / 2) ** 2)
    return h * (eta * gaussian + (1 - eta) * lorentzian)

def scherrer_equation(fwhm, wavelength=1.5406, theta=0.0, shape_factor=0.9):
    theta_rad = np.radians(theta)
    return (shape_factor * wavelength) / (fwhm * np.cos(theta_rad))

def read_xrd_data(file_path):
    data = np.loadtxt(file_path)
    return data[:, 0], data[:, 1]

def baseline_correction(x, y, degree=2):
    p = Polynomial.fit(x, y, degree)
    baseline = p(x)
    return y - baseline

def detect_peaks(x, y):
    y_smooth = savgol_filter(y, window_length=11, polyorder=2)
    prominence = np.max(y_smooth) * 0.015
    peaks, _ = find_peaks(y_smooth, height=np.max(y_smooth) * 0.02, prominence=prominence)
    return peaks, y_smooth

def fit_peaks(x, y, peaks):
    peak_params = []
    
    for peak_idx in peaks:
        peak_x = x[peak_idx]
        peak_y = y[peak_idx]
        
        try:
            popt, _ = curve_fit(pseudo_voigt, x, y, p0=[peak_y, peak_x, 0.2, 0.2, 0.5])
            peak_params.append((peak_idx, popt))
        except:
            continue
    return peak_params

def calculate_goodness_of_fit(y_obs, y_cal):
    weights = 1 / np.maximum(y_obs, 1e-6)
    r_wp = np.sqrt(np.sum(weights * (y_obs - y_cal) ** 2) / np.sum(weights * y_obs ** 2))
    return r_wp


file_path = "CU_B_OBS.txt"
x, y = read_xrd_data(file_path)
y_corrected = baseline_correction(x, y, degree=2)
peaks, y_smooth = detect_peaks(x, y_corrected)

if len(peaks) == 0:
    print("No peaks detected.")
    exit(1)

peak_params = fit_peaks(x, y_corrected, peaks)

plt.figure(figsize=(8, 5))
plt.plot(x, y_corrected, label='Corrected Data', linestyle='dashed')
plt.plot(x, y_smooth, label='Smoothed Data', alpha=0.6)

for peak_idx, popt in peak_params:
    h, x0, sigma, gamma, eta = popt
    fwhm = 2 * sigma * np.sqrt(2 * np.log(2))
    crystallite_size = scherrer_equation(fwhm, theta=x[peak_idx] / 2)
    y_fitted = pseudo_voigt(x, *popt)
    r_wp = calculate_goodness_of_fit(y_corrected, y_fitted)

    print(f"Peak at {x[peak_idx]:.4f}°: Crystallite Size = {crystallite_size:.3f} nm, Goodness of Fit (R_wp) = {r_wp:.3f}")
    
    plt.plot(x, y_fitted, label=f'Fitted Peak at {x[peak_idx]:.4f}°')
    plt.scatter([x[peak_idx]], [h], color='black')

plt.xlabel('2θ (degrees)')
plt.ylabel('Intensity')
plt.legend()
plt.show()
