import os
import json
import sys
from glob import glob

if __name__ == "__main__":
    
    prefix = "A001*"
    output_dir = f"logs/0519_agg"
    os.makedirs(output_dir, exist_ok=True)

    # aggregate loss curves
    files = sorted(glob(f"logs/{prefix}/loss_curve.png"))
    for f in files:
        name = f.split("/")[1]
        os.system(f"cp {f} {output_dir}/{name}_loss_curve.png")
    
    # aggregate videos
    files = glob(f"logs/{prefix}/train.mp4")
    for f in files:
        name = f.split("/")[1]
        os.system(f"cp {f} {output_dir}/{name}_train_video.mp4")
    files = glob(f"logs/{prefix}/test.mp4")
    for f in files:
        name = f.split("/")[1]
        os.system(f"cp {f} {output_dir}/{name}_test_video.mp4")
   
    # create csv
    files = glob(f"logs/{prefix}/metrics.json")
    csv = open(f"{output_dir}/result.csv", "w")
    csv.write("exp_id\tbest_val_loss\tlast_train_loss\tlast_val_loss\n")
    for f in files:
        name = f.split("/")[-2]
        with open(f) as ff:
            lines=ff.readlines()
            try:
                js = json.loads(lines[-1].rstrip())
                print(js)
                csv.write(f"{name}\t{js['best_val_loss']:.3f}\t{js['last_train_loss']:.3f}\t{js['last_val_loss']:.3f}\n")
            except:
                pass
    csv.close()
            
