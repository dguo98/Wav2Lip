import os
import sys
import numpy as np

expid=36
def generate(gpu, bs, lr, n_layer, epochs, lr_p, hidden_dim, wd):
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
    lines[15]=f"HIDDEN_DIM={hidden_dim}\n"
    lines[16]=f"WD={wd:.7f}\n"
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
    hidden_dim=1024
    wd = 0
    
    scripts=[]

    for bs in [8, 32]:
        for lr in [0.001, 0.005]:
            if bs == 32 and lr == 0.001:
                continue
            n_layer = 1
            hidden_dim = 1024
            scripts.append(generate(gpu,bs,lr,n_layer,epochs,lr_p,hidden_dim, wd))    
    for bs in [8, 32]:
        for lr in [0.001]:
            for n_layer in [2,3]:
                for hidden_dim in [64, 256, 1024]:
                    for wd in [0, 1e-2, 1e-4]:
                        scripts.append(generate(gpu,bs,lr,n_layer,epochs,lr_p,hidden_dim,wd))    

    n=3
    for i in range(n):
        f=open(f"bundle_{i}.sh", "w")
        for j,s in enumerate(scripts):
            if j%n==i:
                f.write(f"./{s};")
        f.close()
        os.system(f"chmod +x bundle_{i}.sh")

    
