import os
from shutil import copyfile

direct = "."
folders = sorted(os.listdir(direct))
# print (folders)
for folder in folders:
	print ("folder name", folder)
	if os.path.isdir(folder):
		counter = 1
		os.mkdir("../cacd_faces/"+folder)
		if (os.path.isdir(os.path.join(direct, folder))):
			in_folders = sorted(os.listdir(os.path.join(direct, folder)))
			for in_folder in in_folders:
				if (os.path.isdir(os.path.join(direct, folder+"/"+in_folder))):
					inside_fold = sorted(os.listdir(os.path.join(direct, folder+"/"+in_folder)))
					# print ("inside", inside_fold)
					for file in inside_fold:
						copyfile(os.path.join(direct, folder+"/"+in_folder+"/"+file),"../cacd_faces/"+folder+"/"+str(counter)+".bmp")
						counter += 1
						print ("file", file)