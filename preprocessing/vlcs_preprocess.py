import os, shutil

root_p = "/local/vondrick/chengzhi/VLCS/VLCS/"
root_p = "/local/rcs/mcz/VLCS"

save_p = "/local/vondrick/chengzhi/VLCS_full"
save_p = "/local/rcs/mcz//VLCS_full"
os.makedirs(save_p, exist_ok=True)
for each in os.listdir(root_p):
    d_p = os.path.join(root_p, each, 'full')
    t_p = os.path.join(save_p, each)
    shutil.copytree(d_p, t_p)



