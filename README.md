# LowCostPIV
Software for the use of low-cost PIV set-ups. This code has been developed for the use case of Smoke Image Velocimetry (SIV), but can also be used in conventional PIV applications. Currently implemented tracking methods are:
- [TorchPIV](https://github.com/NikNazarov/TorchPIV), based on [15]
- SIV, based on [6] - the code has been structured to closely resemble TorchPIV
- Optical flow, based on the Horn-Schunck method []

All code is implemented using the PyTorch library, which enables Cuda GPU acceleration. 

Note that for optical flow, **no** pyramidal coarse-to-fine structure is implemented. This makes the method inaccurate for large pixel displacements. The optical flow method by Liu-Shen [...] proves to be more effective.

The SIV method is based on Sum of Absolute Difference (SAD) template matching and could be accelerated with this pyramidal structure.

***
# Example of usage

Running main.py ...
