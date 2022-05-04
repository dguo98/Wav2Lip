import os
import sys
import numpy as np

expid=29
def generate(gpu, bs,warmup,epochs,h,d_model,d_ff,dp,n,seq_len):
    global expid
    expid += 1
    exp_name = f"A{expid:05d}"

    f =open("t_template.sh","r")
    lines = f.readlines()
    f.close()
    lines[2]=f"EXP_ID={exp_name}\n"
    lines[9]=f"GPU={gpu}\n"
    lines[10]=f"BS={bs}\n"
    lines[11] = f"WARMUP={warmup}\n"
    lines[12] = f"EPOCHS={epochs}\n"
    lines[13]=f"H={h}\n"
    lines[14]=f"D_MODEL={d_model}\n"
    lines[15]=f"D_FF={d_ff}\n"
    lines[16]=f"DP={dp}\n"
    lines[17]=f"N={n}\n"
    lines[18]=f"SEQ_LEN={seq_len}\n"

    f = open(exp_name+".sh", "w")
    for line in lines:
        f.write(line)
    f.close()
    os.system(f"chmod +x {exp_name}.sh")
    return f"{exp_name}.sh"

if __name__ == "__main__":
    gpu=0
    bs=8
    warmup=4000
    epochs=20
    h=2
    d_model=512
    d_ff=512
    dp=0.3
    n=5
    seq_len=1

   
    scripts=[]
    bs=32

    for _seq_len in [5, 25]:
        scripts.append(generate(gpu,bs,warmup,epochs,h,d_model,d_ff,dp,n,_seq_len))
    
    for _bs in [64, 128]:
        scripts.append(generate(gpu,_bs,warmup,epochs,h,d_model,d_ff,dp,n,seq_len))

    for _seq_len in [5, 25]:
        _dp=0.4
        scripts.append(generate(gpu,bs,warmup,epochs,h,d_model,d_ff,_dp,n,_seq_len))
           
    n=6
    for i in range(n):
        f=open(f"bundle_{i}.sh", "w")
        for j,s in enumerate(scripts):
            if j%n==i:
                f.write(f"./{s};")
        f.close()
        os.system(f"chmod +x bundle_{i}.sh")

    
