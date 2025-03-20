from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import torch
import yaml
import onnx
from onnxsim import simplify
from sor4onnx import rename
from snc4onnx import combine

class Pre_model(torch.nn.Module):
    def __init__(
        self,
        h: int,
        w: int,
    ):
        super(Pre_model, self).__init__()
        self.h = h
        self.w = w

    def forward(self, x: torch.Tensor):
        x = torch.nn.functional.interpolate(input=x, size=(self.h, self.w))
        return x

class Post_model(torch.nn.Module):
    def __init__(
        self,
        input_h: int,
        input_w: int,
    ):
        super(Post_model, self).__init__()
        self.input_h = input_h
        self.input_w = input_w

    def forward(self, x: torch.Tensor, input_image_bgr: torch.Tensor):
        n, c, h, w = input_image_bgr.shape
        x = torch.nn.functional.interpolate(input=x, size=(h, w))
        return x

def main():
    pyd_H = 512
    pyd_W = 640

    onnx_file = f"pydnet_{pyd_H}x{pyd_W}.onnx"
    output_onnx_file = onnx_file #f"pydnet_512x640_.onnx"
    rename(
        old_new=["encoder", "pydnet/encoder"],
        input_onnx_file_path=onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["decoder", "pydnet/decoder"],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["Resize", "pydnet/Resize"],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["Min", "pydnet/Min"],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["Max", "pydnet/Max"],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["add", "pydnet/add"],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["sub", "pydnet/sub"],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["truediv", "pydnet/truediv"],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    rename(
        old_new=["Relu", "pydnet/Relu"],
        input_onnx_file_path=output_onnx_file,
        output_onnx_file_path=output_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )


    ############### pre-process
    yolo_H=480
    yolo_W=640

    pre_onnx_file = f"preprocess_{yolo_H}x{yolo_W}_{pyd_H}x{pyd_W}.onnx"
    pre_model = Pre_model(h=pyd_H, w=pyd_W)
    x = torch.randn(1, 3, yolo_H, yolo_W).cpu()
    torch.onnx.export(
        pre_model,
        args=(x),
        f=pre_onnx_file,
        opset_version=13,
        input_names=['input_pre'],
        output_names=['output_pre'],
    )
    model_onnx1 = onnx.load(pre_onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, pre_onnx_file)
    onnx_graph = rename(
        old_new=["/", "pydnet/pre/"],
        input_onnx_file_path=pre_onnx_file,
        output_onnx_file_path=pre_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    model_onnx2 = onnx.load(pre_onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, pre_onnx_file)

    ############### post-process
    post_onnx_file = f"postprocess_{yolo_H}x{yolo_W}_{pyd_H}x{pyd_W}.onnx"
    post_model = Post_model(input_h=yolo_H, input_w=yolo_W)
    x = torch.randn(1, 1, pyd_H, pyd_W).cpu()
    y = torch.randn(1, 3, yolo_H, yolo_W).cpu()
    torch.onnx.export(
        post_model,
        args=(x, y),
        f=post_onnx_file,
        opset_version=13,
        input_names=['input_post', 'input_image_bgr'],
        output_names=['depth'],
        dynamic_axes={
            'input_image_bgr' : {2: 'H', 3: 'W'},
            'depth' : {0: '1', 1: '1', 2: 'H', 3: 'W'},
        }
    )
    model_onnx1 = onnx.load(post_onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, post_onnx_file)
    onnx_graph = rename(
        old_new=["/", "pydnet/post/"],
        input_onnx_file_path=post_onnx_file,
        output_onnx_file_path=post_onnx_file,
        mode="full",
        search_mode="prefix_match",
    )
    model_onnx2 = onnx.load(post_onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, post_onnx_file)

    ############### YOLO + PyDNet
    combine(
        srcop_destop = [
            ['output_prep', 'input_pre']
        ],
        input_onnx_file_paths = [
            f'yolov9_e_wholebody34_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
            pre_onnx_file,
        ],
        output_onnx_file_path = f'yolov9_e_wholebody34_with_depth_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
    )
    combine(
        srcop_destop = [
            ['output_pre', 'input_rgb']
        ],
        input_onnx_file_paths = [
            f'yolov9_e_wholebody34_with_depth_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
            f'pydnet_{pyd_H}x{pyd_W}.onnx',
        ],
        output_onnx_file_path = f'yolov9_e_wholebody34_with_depth_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
    )
    rename(
        old_new=["depth", "yolo_depth"],
        input_onnx_file_path=f'yolov9_e_wholebody34_with_depth_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
        output_onnx_file_path=f'yolov9_e_wholebody34_with_depth_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
        mode="outputs",
        search_mode="prefix_match",
    )
    combine(
        srcop_destop = [
            ['yolo_depth', 'input_post', 'input_bgr', 'input_image_bgr'],
        ],
        input_onnx_file_paths = [
            f'yolov9_e_wholebody34_with_depth_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
            post_onnx_file,
        ],
        output_onnx_file_path = f'yolov9_e_wholebody34_with_depth_post_0100_1x3x{yolo_H}x{yolo_W}.onnx',
    )

if __name__ == "__main__":
    main()
