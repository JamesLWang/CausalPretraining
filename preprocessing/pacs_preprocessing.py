import os, shutil

root_p = "/proj/vondrick2/datasets/DGdataset/pacs_data"

list_dir='/proj/vondrick2/datasets/DGdataset/RSC/Domain_Generalization/data/correct_txt_lists'

save_p = "/proj/vondrick2/datasets/DGdataset/pacs_data_val"
os.makedirs(save_p, exist_ok=True)

domain_list=['art_painting', 'cartoon', 'photo', 'sketch']
# for each in domain_list:
#     fname = f'{each}_crossval_kfold.txt'
#     f=open(os.path.join('/proj/vondrick2/datasets/DGdataset/RSC/Domain_Generalization/data/correct_txt_lists', fname), 'r')
#     lines = f.readlines()
#
#     for eachl in lines:
#         imagepath = eachl.split(' ')[0]
#
#         tmp = imagepath
#
#         separate = tmp.split('/')[0]
#         category=tmp.split('/')[1]
#
#         print(imagepath, '\t', separate, '\t', category)
#
#         newfolder = os.path.join(save_p, separate)
#         os.makedirs(newfolder, exist_ok=True)
#         newfolder = os.path.join(save_p, separate, category)
#         os.makedirs(newfolder, exist_ok=True)
#
#         d_p = os.path.join(root_p, imagepath)
#         t_p = os.path.join(save_p, imagepath)
#
#     # d_p = os.path.join(root_p, each, 'full')
#     # t_p = os.path.join(save_p, each)
#         shutil.copyfile(d_p, t_p)

save_p = "/proj/vondrick2/datasets/DGdataset/pacs_data_test"
os.makedirs(save_p, exist_ok=True)
for each in domain_list:
    fname = f'{each}_test_kfold.txt'
    f = open(os.path.join('/proj/vondrick2/datasets/DGdataset/RSC/Domain_Generalization/data/correct_txt_lists', fname), 'r')
    lines = f.readlines()

    for eachl in lines:
        imagepath = eachl.split(' ')[0]

        tmp = imagepath

        separate = tmp.split('/')[0]
        category=tmp.split('/')[1]

        print(imagepath, '\t', separate, '\t', category)

        newfolder = os.path.join(save_p, separate)
        os.makedirs(newfolder, exist_ok=True)
        newfolder = os.path.join(save_p, separate, category)
        os.makedirs(newfolder, exist_ok=True)

        d_p = os.path.join(root_p, imagepath)
        t_p = os.path.join(save_p, imagepath)

    # d_p = os.path.join(root_p, each, 'full')
    # t_p = os.path.join(save_p, each)
        shutil.copyfile(d_p, t_p)

