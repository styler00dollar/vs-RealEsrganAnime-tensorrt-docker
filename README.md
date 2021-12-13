# vs-RealEsrganAnime-tensorrt-docker

Using image super resolution models with vapoursynth and speeding them up with TensorRT. Also a docker image since TensorRT is hard to install. Testing showed ~70% more speed on my 1070ti compared to normal PyTorch in 480p. Using the 2x model with TensorRT and 848x480 input was 0.517x realtime speed for 24fps video.

I was forced to use [onnx/onnx-tensorrt](https://github.com/onnx/onnx-tensorrt) instead of [NVIDIA/Torch-TensorRT](https://github.com/NVIDIA/Torch-TensorRT) because of convertion errors with PyTorch, but the only disadvantage should be that a new onnx model needs to be created for a different input resolution, which takes a bit time.

This repo uses a lot of code from [HolyWu/vs-realesrgan](https://github.com/HolyWu/vs-realesrgan) and [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). The models are from [here](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.3.0).

Usage:
```
# install docker, command for arch
yay -S docker nvidia-docker nvidia-container-toolkit
# Put the dockerfile in a directory and run that inside that directory
docker build -t realsr_tensorrt:latest .
# run with a mounted folder
docker run --privileged --gpus all -it --rm -v /home/Desktop/tensorrt:/workspace/tensorrt realsr_tensorrt:latest
# you can use it in various ways, ffmpeg example
vspipe --y4m inference.py - | ffmpeg -i pipe: example.mkv
```

If you don't want to use docker, vapoursynth install commands are [here](https://github.com/styler00dollar/vs-vfi) and a TensorRT example is [here](https://github.com/styler00dollar/Colab-torch2trt/blob/main/Colab-torch2trt.ipynb).

You can choose between the 4x and 2x model inside of `inference.py`.
