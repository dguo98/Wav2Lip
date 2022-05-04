import numpy as np
import os
import sys
from glob import glob
from tqdm import tqdm
from IPython import embed

if __name__ == "__main__":
    folders = glob("s*_info*")
    for folder in folders:
        root_id = os.path.basename(folder).replace("_info", "")
        root_path = f"{root_id}_info"
        print("root_path=", root_path)
        info_paths = sorted(glob(f"{root_path}/*.csv"))
        def get_id(path):
            return path.split("/")[-1].replace(".csv", "").split("_")[1]
        
        # convert to csv
        f = open(f"{root_id}.csv", "w") 
        max_id = len(glob(f"{root_id}_info/*.csv"))
        last_line = None
        for i in tqdm(range(max_id)):
            info_path = f"{root_path}/frame_{i:06d}.csv"
            if not os.path.exists(info_path):
                print("not found", info_path)
                assert last_line is not None and i != 0
                f.write(f"{i:06d}," + last_line)
            else:
                ff=open(info_path)
                lines=ff.readlines()
                ff.close()

                if i==0:
                    f.write("id," + lines[0])
                f.write(f"{i:06d}," + lines[1])
                last_line = lines[1]
        f.close()

        # convert to vecs
        vecs = []
        f = open(f"{root_id}.csv", "r")
        lines = f.readlines()
        f.close()
        assert len(lines) == max_id+1
        for line in lines[1:]:
            nums = [float(t) for t in line.rstrip().split(",")]
            vecs.append(np.array(nums))
        vecs = np.stack(vecs, axis=0)
        print("vecs.shape=",vecs.shape)
        np.save(f"{root_id}_openface.npy", vecs)


