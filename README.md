# image-processing

### Overview

This project provides a robust command-line interface (CLI) for performing various image processing operations. Leveraging popular Python libraries like `OpenCV`, `Pillow`, `NumPy`, `SciPy`, `Matplotlib`, and `Pandas`, it enables you to apply filters, dithering, contrast adjustments, Fourier analysis, and resizing techniques directly from your terminal.

### Requirements

 - Python 3.10+
 - pip

### Download

Download the source code using the ```git clone``` command:

```bash
$ git clone https://github.com/wedkarz02/image-processing.git
```

Or use the *Download ZIP* option from the Github repository [page](https://github.com/wedkarz02/image-processing.git).

### Quick Setup

Create a virtual environment:

```bash
$ python3 -m venv venv
```
You might need to install the ```venv``` package in order to do so.

Install required packages in the virtual environment from the ```requirements.txt``` file:

```bash
$ venv/bin/pip3 install -r requirements.txt
```

### Usage

```
usage: main.py [-h]
               {floyd-steinberg,jarvis-judice-ninke,ordered-dithering,ordered-dithering-bayer-8x8,sine-window,averaging-filter,k-trimmed-mean,k-nearest-neighbor,symmetric-nearest-neighbor,gamma-correction,van-cittert-deconvolution,histogram-equalization,hyperbolic-histogram-equalization,contrast-lut,apply-color-lut,pointillism,global-contrast,local-contrast,fourier-analysis,plot-histogram,plot-profile,zhang-suen-thinning,otsu-thresholding,iterative-three-class-thresholding,local-otsu-thresholding,resize-nearest-neighbor,resize-average-two-horizontal,resize-bilinear,resize-rgb-min-max-avg} ...

Apply various image processing and analysis techniques.

positional arguments:
  {floyd-steinberg,jarvis-judice-ninke,ordered-dithering,ordered-dithering-bayer-8x8,sine-window,averaging-filter,k-trimmed-mean,k-nearest-neighbor,symmetric-nearest-neighbor,gamma-correction,van-cittert-deconvolution,histogram-equalization,hyperbolic-histogram-equalization,contrast-lut,apply-color-lut,pointillism,global-contrast,local-contrast,fourier-analysis,plot-histogram,plot-profile,zhang-suen-thinning,otsu-thresholding,iterative-three-class-thresholding,local-otsu-thresholding,resize-nearest-neighbor,resize-average-two-horizontal,resize-bilinear,resize-rgb-min-max-avg}
                        Available image processing commands. Use '[command] --help' for command-specific options.
    floyd-steinberg     Apply Floyd-Steinberg error diffusion dithering.
    jarvis-judice-ninke
                        Apply Jarvis-Judice-Ninke error diffusion dithering.
    ordered-dithering   Apply ordered dithering using a generated Bayer matrix.
    ordered-dithering-bayer-8x8
                        Apply ordered dithering using a fixed 8x8 Bayer matrix with explicit rules.
    sine-window         Apply a sine window filter.
    averaging-filter    Apply an averaging (blur) filter.
    k-trimmed-mean      Apply K-trimmed mean filter.
    k-nearest-neighbor  Apply K-nearest neighbor filter.
    symmetric-nearest-neighbor
                        Apply Symmetric Nearest Neighbor filter.
    gamma-correction    Apply gamma correction to match image brightness.
    van-cittert-deconvolution
                        Apply Van Cittert deconvolution to a blurred image.
    histogram-equalization
                        Apply standard histogram equalization.
    hyperbolic-histogram-equalization
                        Apply hyperbolic histogram equalization.
    contrast-lut        Apply a contrast Look-Up Table (LUT).
    apply-color-lut     Apply a specific custom color LUT.
    pointillism         Apply a pointillism effect.
    global-contrast     Calculate global contrast of an image.
    local-contrast      Calculate local contrast of an image.
    fourier-analysis    Perform Fourier Transform analysis and plot results.
    plot-histogram      Plot histogram and CDF of an image.
    plot-profile        Plot image profile from CSV with sampling points.
    zhang-suen-thinning
                        Apply Zhang-Suen thinning algorithm.
    otsu-thresholding   Apply Otsu's global thresholding.
    iterative-three-class-thresholding
                        Apply iterative three-class thresholding.
    local-otsu-thresholding
                        Apply local Otsu's thresholding.
    resize-nearest-neighbor
                        Resize image using Nearest Neighbor interpolation.
    resize-average-two-horizontal
                        Resize image by averaging two horizontal pixels.
    resize-bilinear     Resize image using Bilinear interpolation.
    resize-rgb-min-max-avg
                        Resize RGB image using custom min/max average method.

options:
  -h, --help            show this help message and exit
```

### License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/wedkarz02/image-processing/blob/main/LICENSE) file for more info.