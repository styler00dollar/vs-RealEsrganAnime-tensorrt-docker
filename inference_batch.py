import sys
import os
sys.path.append('/workspace/tensorrt/')
from src.vs import RealESRGAN
import vapoursynth as vs

tmp_dir = "tmp/"
core = vs.core
core.num_threads = 16
core.std.LoadPlugin(path='/usr/lib/x86_64-linux-gnu/libffms2.so')
with open(os.path.join(tmp_dir, "tmp.txt")) as f:
    txt = f.readlines()
clip = core.ffms2.Source(source=txt)
# convert colorspace
#clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s='709')
# convert colorspace + resizing
clip = vs.core.resize.Bicubic(clip, width=848, height=480, format=vs.RGBS, matrix_in_s='709')
# add scale param here to set the different scale. 2 and 4 are the possible values. 2 Is default
clip = RealESRGAN(clip, scale=2, fp16=False)
clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
clip.set_output()
