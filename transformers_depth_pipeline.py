import argparse
import cv2
from tqdm import tqdm
import os
import numpy as np
from transformers import pipeline
from demo import load_video_paths
from demo import write_img
from demo import color

def run(args):
    estimator = pipeline(
        task = "depth-estimation",
        model = "Intel/dpt-large",
    )
    # get input
    path_lists, scene_names, dict_output_names = load_video_paths(args)
   
    for i in range(len(path_lists)):
        output_list = []
        for f in tqdm(path_lists[i]):
            # Apply prediction
            result = estimator(f)
            depth = result["depth"]

            output_list.append(np.array(depth))
        
        output_name = os.path.join(args.output, scene_names[i])
        
        write_img(
            model_name=args.model_name,
            filename=str(output_name),
            output_list=output_list,
            output_names=dict_output_names[scene_names[i]],
            is_colored=False
            )
        
        color_list = color(output_list)
        write_img(
            model_name=args.model_name,
            filename=str(output_name),
            output_list=color_list,
            output_names=dict_output_names[scene_names[i]],
            is_colored=True
            )
    return
if __name__ == "__main__":
    print("hello")
    parser = argparse.ArgumentParser(description = "Depth Estimation")
    parser.add_argument('--input', default = './input', type = str, help = 'video root path')
    parser.add_argument('--output', default='./output', type = str, help = 'path to save output')
    parser.add_argument('--model_name', default= "transformers", type=str)

    args = parser.parse_args()
    print("Run")
    run(args)
    print("Finished")
# Example:
# python test-model1.py --input ./videos
