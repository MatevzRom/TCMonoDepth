#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import os

import cv2
import torch
from torchvision.transforms import Compose

from networks.transforms import Resize
from networks.transforms import PrepareForNet
from tqdm import tqdm

def write_img(model_name: str,filename: str, output_list, output_names, is_colored: bool):
    assert (len(output_list) > 0)
    filename += "_" + model_name + "_result"

    # prepare output folder
    if is_colored:
        filename += "_colored"
    os.makedirs(filename, exist_ok=True)

    for index, img in enumerate(output_list):
        cv2.imwrite(filename +"/"+ output_names[index], img)
    return

def process_depth(dep):
    dep = dep - dep.min()
    dep = dep / dep.max()
    dep_vis = dep * 255

    return dep_vis.astype('uint8')


def load_video_paths(args):
    root_path = args.input
    output_path = args.output
    scene_names = sorted(os.listdir(root_path))
    
    path_lists = []
    dict_output_names = {}
    ignored= {"all_file.txt", "bg_img.txt", ".directory"}
    for scene in scene_names:
        frame_names = sorted([x for x in os.listdir(os.path.join(root_path, scene)) if x not in ignored])
        frame_paths = [os.path.join(root_path, scene, name) for name in frame_names]
        path_lists.append(frame_paths)
        dict_output_names[scene] = frame_names

    return path_lists, scene_names, dict_output_names

def color(output_list):
    color_list = []
    for j in range(len(output_list)):
        frame_color = cv2.applyColorMap(output_list[j], cv2.COLORMAP_INFERNO)
        color_list.append(frame_color)
    return color_list

def run(args):
    print("Initialize")

    # select device
    device = torch.device("cuda")
    print("Device: %s" % device)

    # load network
    print("Creating model...")
    if args.model == 'large':
        from networks import MidasNet
        model = MidasNet(args)
    else:
        from networks import TCSmallNet
        model = TCSmallNet(args)

    if os.path.isfile(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location='cpu'))
        print("Loading model from " + args.resume)
    else:
        print("Loading model path fail, model path does not exists.")
        exit()

    model.cuda().eval()
    print("Loading model done...")

    transform = Compose([
        Resize(
            args.resize_size,  #width
            args.resize_size,  #height
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        PrepareForNet(),
    ])

    # get input
    path_lists, scene_names, dict_output_names = load_video_paths(args)

    # prepare output folder
    os.makedirs(args.output, exist_ok=True)

    for i in range(len(path_lists)):
        print("Prcessing: %s" % scene_names[i])
        img0 = cv2.imread(path_lists[i][0])
        # predict depth
        output_list = []
        with torch.no_grad():
            for f in tqdm(path_lists[i]):
                frame = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
                frame = transform({"image": frame})["image"]
                frame = torch.from_numpy(frame).to(device).unsqueeze(0)

                prediction = model.forward(frame)
                print(prediction.min(), prediction.max())
                prediction = (torch.nn.functional.interpolate(
                    prediction,
                    size=img0.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy())
                output_list.append(prediction)

        # save output
        # output_name without file type ending
        output_name = os.path.join(args.output, scene_names[i])
        
        output_list = [process_depth(out) for out in output_list]

        # Writes grayscale output

        write_img(args.model_name, output_name, output_list, dict_output_names[scene_names[i]], False)
    
        # Writes colored output
        color_list = color(output_list)
        write_img(args.model_name, output_name, color_list, dict_output_names[scene_names[i]], True)

    print(args.output + " Done.")

if __name__ == "__main__":
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Settings
    parser = argparse.ArgumentParser(description="A PyTorch Implementation of Video Depth Estimation")

    parser.add_argument('--model', default='large', choices=['small', 'large'], help='size of the model')
    parser.add_argument('--resume', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--input', default='./input', type=str, help='video root path')
    parser.add_argument('--output', default='./output', type=str, help='path to save output')
    parser.add_argument('--model_name', default= "TCMD", type=str)
    parser.add_argument('--resize_size',
                        type=int,
                        default=384,
                        help="spatial dimension to resize input (default: small model:256, large model:384)")
    parser

    args = parser.parse_args()

    print("Run")
    run(args)

# Example:
# python demo.py --model large --resume ./weights/_ckpt.pt.tar --input ./videos --output ./output --resize_size 384