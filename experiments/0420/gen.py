import os
import sys
import numpy as np

expid=0
def generate(gpu, bs, lr, n_layer, epochs, lr_p):
    global expid
    expid += 1
    exp_name = f"B{expid:05d}"

    f =open("template.sh","r")
    lines = f.readlines()
    f.close()
    lines[2]=f"EXP_ID={exp_name}\n"
    lines[9]=f"GPU={gpu}\n"
    lines[10]=f"BS={bs}\n"
    lines[11]=f"LR={lr:.5f}\n"
    lines[12]=f"LR_P={lr_p}\n"
    lines[13]=f"N_LAYER={n_layer}\n"
    lines[14]=f"EPOCHS={epochs}\n"
    f = open(exp_name+".sh", "w")
    for line in lines:
        f.write(line)
    f.close()
    os.system(f"chmod +x {exp_name}.sh")
    return f"{exp_name}.sh"

if __name__ == "__main__":
    gpu=0
    bs=16
    lr=0.005
    n_layer=1
    epochs=100
    lr_p=10
    
    scripts=[]
    for bs in [8, 16, 32, 64]:
        for lr in [0.001, 0.005, 0.01]:
            for n_layer in [1, 2, 3]:
                scripts.append(generate(gpu,bs,lr,n_layer,epochs,lr_p))    

    n=3
    for i in range(n):
        f=open(f"bundle_{i}.sh", "w")
        for j,s in enumerate(scripts):
            if j%n==i:
                f.write(f"./{s};")
        f.close()
        os.system(f"chmod +x bundle_{i}.sh")

    
