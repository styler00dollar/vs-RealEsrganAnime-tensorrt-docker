# Code mainly from https://github.com/HolyWu/vs-realesrgan
import vapoursynth as vs
import onnx as ox
import onnx_tensorrt.backend as backend
import os
import numpy as np
import torch
from src.SRVGGNetCompact import SRVGGNetCompact
core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4

def RealESRGAN(clip: vs.VideoNode, scale: int = 2, fp16: bool = False) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('RealESRGAN: this is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('RealESRGAN: only RGBS format is supported')

    if scale not in [2, 4]:
        raise vs.Error('RealESRGAN: scale must be 2 or 4')
    
    # load network
    model_path = f'/workspace/RealESRGANv2-animevideo-xsx{scale}.pth'
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=scale, act_type='prelu')
    model.load_state_dict(torch.load(model_path, map_location="cpu")['params'])
    model.eval()
    # export to onnx and load with tensorrt (you cant use https://github.com/NVIDIA/Torch-TensorRT because the scripting step will fail)
    torch.onnx.export(model, (torch.rand(1,3,clip.height,clip.width)), f"/workspace/test.onnx", verbose=False, opset_version=13)
    model = ox.load("/workspace/test.onnx")
    model = backend.prepare(model, device='CUDA:0', fp16_mode=fp16)

    def realesrgan(n, f):
        img = frame_to_tensor(f[0])
        output = model.run(img)[0]
        return tensor_to_frame(output, f[1].copy())

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale)
    return new_clip.std.ModifyFrame(clips=[clip, new_clip], selector=realesrgan)


def frame_to_tensor(f):
    arr = np.stack([np.asarray(f.get_read_array(plane) if vs_api_below4 else f[plane]) for plane in range(f.format.num_planes)])
    arr = np.expand_dims(arr, 0)
    return arr


def tensor_to_frame(t, f):
    arr = np.squeeze(t, 0)
    arr = np.clip(arr, 0, 1)
    for plane in range(f.format.num_planes):
        np.copyto(np.asarray(f.get_write_array(plane) if vs_api_below4 else f[plane]), arr[plane, :, :])
    return f