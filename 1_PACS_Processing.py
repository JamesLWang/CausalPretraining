import glob, os

all_images = sorted(glob.glob("/proj/vondrick/james/OfficeHomeDataset_10072016/*"))
domain_names = [os.path.basename(x) for x in all_images]
domain_names

classes = sorted(glob.glob(os.path.join(domain, "*")))
class_names = [os.path.basename(x) for x in classes]

outputDir = "/proj/vondrick/james/OfficeHomeDataset_Category_Final"
for class_name in class_names:
    newClassFile = os.path.join(outputDir, class_name) 
    isExist = os.path.exists(newClassFile)
    if not isExist:
        os.makedirs(newClassFile)
    for domain_name in domain_names:
        newDomainFile = os.path.join(newClassFile, domain_name)
        isExist = os.path.exists(newDomainFile)
        if not isExist:
            os.makedirs(newDomainFile)
            
import shutil

for domain in all_images:
    classes = sorted(glob.glob(os.path.join(domain, "*")))
    class_names = [os.path.basename(x) for x in classes]
    for class_ in classes:
        imgs = sorted(glob.glob(os.path.join(class_, "*")))
        ctr = 0
        for img in imgs:
            dst = os.path.join(outputDir, os.path.basename(class_), os.path.basename(domain), str(ctr) + ".png")
            src = img
            shutil.copyfile(src, dst)
            ctr += 1
            
            