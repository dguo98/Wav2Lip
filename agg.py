import os
import json
import sys
from glob import glob

if __name__ == "__main__":
    
    prefix = "B001*"
    output_dir = f"logs/0427_agg"
    os.makedirs(output_dir, exist_ok=True)

    # aggregate loss curves
    files = sorted(glob(f"logs/{prefix}/loss_curve.png"))
    for f in files:
        name = f.split("/")[1]
        os.system(f"cp {f} {output_dir}/{name}_loss_curve.png")
    
    # aggregate videos
    files = glob(f"logs/{prefix}/inference/predict_with_audio.mp4")
    for f in files:
        name = f.split("/")[1]
        os.system(f"cp {f} {output_dir}/{name}_video.mp4")
    files = glob(f"logs/{prefix}/tf_inference/predict_with_audio.mp4")
    for f in files:
        name = f.split("/")[1]
        os.system(f"cp {f} {output_dir}/{name}_tfvideo.mp4")
    
    # create csv
    files = glob(f"logs/{prefix}/metrics.json")
    csv = open(f"{output_dir}/result.csv", "w")
    csv.write("exp_id\tbest_val_loss\tlast_train_loss\tlast_val_loss\n")
    for f in files:
        name = f.split("/")[-2]
        with open(f) as ff:
            js = json.load(ff)
            csv.write(f"{name}\t{js['best_val_loss']:.3f}\t{js['last_train_loss']:.3f}\t{js['last_val_loss']:.3f}\n")
    csv.close()
            
