# LowCostPIV
Software for the use of low-cost PIV set-ups, based on [TorchPIV](https://github.com/NikNazarov/TorchPIV). This code has been developed for the use case of Smoke Image Velocimetry (SIV), but can also be used in conventional PIV applications. All code is implemented using the [PyTorch](https://github.com/pytorch) library, which enables Cuda GPU acceleration. 

Currently implemented tracking methods are:
- Template matching: cross-correlation and Sum of Absolute Difference (SAD)
- Optical flow, based on the Horn-Schunck method

Both methods have a multipass mode for large displacements.

## Example of usage

Running `main.py`

```python
video_file = "Cilinder.MOV"
fn = video_file.split(".")[0]

# reference frame specified first, then the range we want to analyse with SIV
frames = [0, *(i for i in range(225, 325))]

vid = Video(rf"Test Data/{video_file}", df='.png', indices=frames)
vid.create_frames()

processor = Processor(rf"Test Data/{fn}", df='.png')
processor.postprocess()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

capture_fps = 240.
scale = 0.02

siv = SIV(
    folder=rf"Test Data/{fn}_PROCESSED",
    device=device,
    window_size=64,
    overlap=32,
    search_area=(20, 20, 20, 20)
)

if f"{fn}_RESULTS.npy" not in os.listdir(f"Test Data"):
    x, y, vx, vy = siv.run(mode=1)
    res = np.array((x.cpu(), y.cpu(), vx.cpu(), vy.cpu()))
    np.save(rf"Test Data/{fn}_RESULTS", res)
else:
    print("Loading results...")
    res = np.load(rf"Test Data/{fn}_RESULTS.npy")

viewer = Viewer(rf"Test Data/{fn}_PROCESSED", playback_fps=30., capture_fps=capture_fps)

# viewer.play_video()
viewer.vector_field(res, scale)
# viewer.velocity_field(res, scale, 30, 'cubic')
```

The script runs through the following steps:
- Capture video frames from a specified path 
- Pre-process the video frames (background removal)
- Run SIV algorithm
- Plot resulting velocity field

The input of the script is a smoke plume simulation. The result is shown below
...
