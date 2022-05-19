import os
import sys
import numpy as np

expid=97
def generate(gpu, bs,warmup,epochs,h,d_model,d_ff,dp,n,seq_len,use_pose,
    img_type, img_loss, img_mouth, lmk_loss, perc_loss, template="template"):
    global expid
    expid += 1
    exp_name = f"A{expid:05d}"
    
    f =open(f"{template}.sh","r")
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
    lines[19]=f"USE_POSE={use_pose}\n"
    lines[20]=f"IMG_TYPE={img_type}\n"
    lines[21]=f"IMG_LOSS={img_loss:.3f}\n"
    lines[22]=f"IMG_SIZE=256\n"  # NB(demi): fix to 256 for now
    lines[25]=f"IMG_MOUTH={img_mouth}\n"
    lines[26] = f"LMK_LOSS={lmk_loss:.3f}\n"
    lines[27] = f"PERC_LOSS={perc_loss:.3f}\n"


    f = open(exp_name+".sh", "w")
    for line in lines:
        f.write(line)
    f.close()
    os.system(f"chmod +x {exp_name}.sh")
    return f"{exp_name}.sh"

if __name__ == "__main__":
    gpu=0
    bs=32
    warmup=4000
    epochs=30
    h=2
    d_model=512
    d_ff=512
    dp=0.3
    n=5
    seq_len=1
    use_pose=0
    img_type="gt"
    img_loss=0.0
    img_mouth=1
    lmk_loss=0.0
    perc_loss=0.0
    
    scripts = []

    # only latent loss
    for _img_loss in [0.0]:
        for _bs in [96, 128]:
            scripts.append(generate(gpu,_bs,warmup,epochs,h,d_model,d_ff,dp,n,seq_len, use_pose,"none",_img_loss,img_mouth, lmk_loss, perc_loss))

    for _d_ff in [1024, 2048]:
        for _bs in [32]:
            scripts.append(generate(gpu,_bs,warmup,epochs,h,d_model,_d_ff,dp,n,seq_len, use_pose,"none",img_loss,img_mouth, lmk_loss, perc_loss))
    
    # seq length > 1
    for _seq_len in [5, 25]:
        _bs =32
        scripts.append(generate(gpu,_bs,warmup,epochs,h,d_model,d_ff,dp,n,_seq_len, use_pose,"none",img_loss,img_mouth, lmk_loss, perc_loss))

    # finetune mouth pixel loss
    for _img_loss in [0.2, 0.5]:
        for _bs in [4]:
            _epochs=10
            scripts.append(generate(gpu,_bs,warmup,_epochs,h,d_model,d_ff,dp,n,seq_len, use_pose,"gt",_img_loss,img_mouth, lmk_loss, perc_loss, "load_template"))


    
    n=2
    for i in range(n):
        f=open(f"bundle_{i}.sh", "w")
        for j,s in enumerate(scripts):
            if j%n==i:
                f.write(f"./{s};")
        f.close()
        os.system(f"chmod +x bundle_{i}.sh")

    
