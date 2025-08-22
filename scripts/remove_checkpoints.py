import os


path = "/root/BHRL/work_dirs/vot/BHRL/first_images_seperate/"

for (root,dirs,files) in os.walk(path):
   for file in files:
      if file.endswith(".pth") and not "latest" in file:
         epoch = int(file[file.find("epoch")+6:file.find(".")])
         if (epoch % 50):
            print(os.path.join(root,file))
            os.remove(os.path.join(root,file))