# LowCostPIV
Software for the use of low-cost PIV set-ups, based on [TorchPIV](https://github.com/NikNazarov/TorchPIV). This code has been developed for the use case of Smoke Image Velocimetry (SIV), but can also be used in conventional PIV applications. All code is implemented using the [PyTorch](https://github.com/pytorch) library, which enables Cuda GPU acceleration. 

Currently implemented tracking methods are:
- Template matching: cross-correlation and Sum of Absolute Difference (SAD)
- Optical flow, based on the Horn-Schunck method
- 
Both methods have a pyramidal multipass mode.

## Example of usage

Running `main.py` ...
