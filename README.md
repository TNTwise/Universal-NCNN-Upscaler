# Upscaling-Models-NCNN
## A repo of models converted to NCNN
 - Model conversion types supported:
 - ESRGAN (ESRGAN, ESRGAN+, "new-arch ESRGAN" (RealSR, BSRGAN), SPSR, and Real-ESRGAN) Models. Converted by <a href="https://github.com/chaiNNer-org/chaiNNer">chaiNNer</a>
 - Compact Models
 - SPAN Models
 - MSDAN Models
 - All models belong to their respecitve owners under their respective licenses. Please submit an issue if you do not want your model posted here.
## General Models:
 - <a href="https://github.com/TNTwise/Upscaling-Models-NCNN/releases/tag/General">Download NCNN</a>
 - <a href="https://github.com/cszn/KAIR/releases/tag/v1.0">BSRGAN (2X)</a>
## Realistic Models:  
 - <a href="https://github.com/TNTwise/Upscaling-Models-NCNN/releases/tag/Realistic">Download NCNN</a>
 - <a href="https://openmodeldb.info/models/4x-LSDIR">4xLSDIR (4X)</a> by Helaman
 - <a href="https://openmodeldb.info/models/4x-LSDIRPlus">4xLSDIRPlus (4X)</a> by Helaman
 - <a href="https://openmodeldb.info/models/4x-LSDIRPlusR">4xLSDIRPlusR (4X)</a> by Helaman
 - <a href="https://openmodeldb.info/models/4x-LSDIRPlusC">4xLSDIRPlusC (4X)</a> by Helaman
 - <a href="https://openmodeldb.info/models/4x-LSDIRPlusN">4xLSDIRPlusN (4X)</a> by Helaman
 - <a href="https://openmodeldb.info/models/4x-Nomos8kSC">Nomos8kSC (4X)</a> by Helaman
 - <a href="https://openmodeldb.info/models/4x-NMKD-Siax-CX">NMKD Siax (4X)</a> by NMKD
 - <a href="https://openmodeldb.info/models/4x-ClearRealityV1">ClearRealityV1 (4X)</a> by Kim2091
 - <a href="https://openmodeldb.info/models/4x-ClearRealityV1">ClearRealityV1-Soft (4X)</a> by Kim2091
## Animation
 - <a href="https://github.com/TNTwise/Upscaling-Models-NCNN/releases/tag/Animation">Download NCNN</a>
 - <a href="https://openmodeldb.info/models/2x-sudo-RealESRGAN">sudo-RealESRGAN (2X)</a> by styler00dollar/sudo
 - <a href="https://openmodeldb.info/models/2x-sudo-shuffle-cugan-9-584-969">sudo-shuffle-CUGAN (2X)</a> by styler00dollar/sudo
 - <a href="https://openmodeldb.info/models/2x-AniScale-2-Compact">AniScale-2-Compact (2X)</a> by Sirosky
 - <a href="https://openmodeldb.info/models/2x-AnimeJaNai-v2-Compact">AnimeJaNai-v2-Compact (2X)</a> by the database








## Usages
### Based on upscayl-bin.
Input one image, output one upscaled frame image.<br/>
Place bin/param file in models folder, then use command to upscale.
### Example Commands

```shell
./upscayl-bin -m models/ -n 4xLSDIR -s 4 -i 0.jpg  -o 01.jpg
./upscayl-bin -m models/ -n 4xLSDIR -s 4 -i input_frames/ -o output_frames/
```

Example below runs on CPU, Discrete GPU, and Integrated GPU all at the same time. Uses 2 threads for image decoding, 4 threads for one CPU worker, 4 threads for another CPU worker, 2 threads for discrete GPU, 1 thread for integrated GPU, and 4 threads for image encoding.
```shell
./upscayl-bin -m models/ -n 4xLSDIR -s 4 -i input_frames/ -o output_frames/ -g -1,-1,0,1 -j 2:4,4,2,1:4
```

### Video Upscaling with FFmpeg

```shell
mkdir input_frames
mkdir output_frames

# find the source fps and format with ffprobe, for example 24fps, AAC
ffprobe input.mp4

# extract audio
ffmpeg -i input.mp4 -vn -acodec copy audio.m4a

# decode all frames
ffmpeg -i input.mp4 input_frames/frame_%08d.png

# upscale 4x resolution
./upscayl-bin -m models/ -n 4xLSDIR -s 4 -i input_frames -o output_frames

# encode interpolated frames in 48fps with audio
ffmpeg -framerate 24 -i output_frames/%08d.png -i audio.m4a -c:a copy -crf 20 -c:v libx264 -pix_fmt yuv420p output.mp4
```

### Full Usages

```console
Usage: upscayl-bin -i infile -o outfile [options]...

  -h                   show this help
  -i input-path        input image path (jpg/png/webp) or directory
  -o output-path       output image path (jpg/png/webp) or directory
  -s scale             upscale ratio (can be 2, 3, 4. default=4)
  -t tile-size         tile size (>=32/0=auto, default=0) can be 0,0,0 for multi-gpu
  -m model-path        folder path to the pre-trained models. default=models
  -n model-name        model name (default=4xLSDIR, can be 4xLSDIR | spanx2_ch52 | 4xLSDIR | spanx4_ch52)
  -g gpu-id            gpu device to use (default=auto) can be 0,1,2 for multi-gpu
  -c cpu-only          use only CPU for upscaling, instead of vulkan
  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu
  -x                   enable tta mode
  -f format            output image format (jpg/png/webp, default=ext/png)
  -v                   verbose output
```

- `input-path` and `output-path` accept file directory
- `load:proc:save` = thread count for the three stages (image decoding + upscaling + image encoding), using larger values may increase GPU usage and consume more GPU memory. You can tune this configuration with "4:4:4" for many small-size images, and "2:2:2" for large-size images. The default setting usually works fine for most situations. If you find that your GPU is hungry, try increasing thread count to achieve faster processing.
- `pattern-format` = the filename pattern and format of the image to be output, png is better supported, however webp generally yields smaller file sizes, both are losslessly encoded
- `scale` = upscale multiplier, must match model.

If you encounter a crash or error, try upgrading your GPU driver:

- Intel: https://downloadcenter.intel.com/product/80939/Graphics-Drivers
- AMD: https://www.amd.com/en/support
- NVIDIA: https://www.nvidia.com/Download/index.aspx



## Other Open-Source Code Used

- https://github.com/Tencent/ncnn for fast neural network inference on ALL PLATFORMS
- https://github.com/webmproject/libwebp for encoding and decoding Webp images on ALL PLATFORMS
- https://github.com/nothings/stb for decoding and encoding image on Linux / MacOS
- https://github.com/tronkko/dirent for listing files in directory on Windows
