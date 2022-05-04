import os
import sys
import numpy as np

expid=112
def generate(gpu, bs, lr, n_layer, epochs, lr_p, hidden_dim, wd,dp, seq_len):
    global expid
    expid += 1
    exp_name = f"B{expid:05d}"

    f =open("conv_template.sh","r")
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
    lines[17]=f"DP={dp:.2f}\n"
    lines[18]=f"SEQ_LEN={seq_len}\n"
    f = open(exp_name+".sh", "w")
    for line in lines:
        f.write(line)
    f.close()
    os.system(f"chmod +x {exp_name}.sh")
    return f"{exp_name}.sh"

if __name__ == "__main__":
    gpu=0
    bs=16
    lr=0.001
    n_layer=1
    epochs=100
    lr_p=10
    hidden_dim=512
    wd = 0
    dp=0
    seq_len=1
    
    scripts=[]
    for seq_len in [1, 2, 8]:
        for n_layer in [1, 2]:
            for bs in [32, 64]:
                scripts.append(generate(gpu,bs,lr,n_layer,epochs,lr_p,hidden_dim,wd,dp, seq_len))    


    n=1
    for i in range(n):
        f=open(f"bundle_{i}.sh", "w")
        for j,s in enumerate(scripts):
            if j%n==i:
                f.write(f"./{s};")
        f.close()
        os.system(f"chmod +x bundle_{i}.sh")

    
