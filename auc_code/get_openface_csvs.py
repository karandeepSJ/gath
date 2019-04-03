import os
import csv
from shutil import copyfile

direct = "."
folders = sorted(os.listdir(direct))
required = [" AU01_c", " AU02_c", " AU04_c"," AU05_c"," AU06_c"," AU07_c"," AU09_c"," AU10_c"," AU12_c"," AU14_c"," AU15_c"," AU17_c"," AU20_c"," AU23_c"," AU25_c"," AU26_c"," AU28_c"," AU45_c"]
print (len(required))

# print (folders)
for folder in folders:
	print ("folder name", folder)
	if os.path.isdir(folder):
		os.mkdir("../auc_ravdess_csv/"+folder)
		if (os.path.isdir(os.path.join(direct, folder))):
			in_folders = sorted(os.listdir(os.path.join(direct, folder)))
			counter = 1
			for file in in_folders:
				if file[-3:] == "csv":
					write_list = []
					with open(folder+"/"+file, newline='') as csvfile:
						linereader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
						for row in linereader:
							for each in required:
								write_list.append(row[each])
					if (len(write_list) > 18):
						print ("exceeding", folder, file)
					else:
						with open("../auc_ravdess_csv/"+folder+"/"+str(counter)+".csv", mode='w') as csv_file:
							writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
							writer.writerow(required)
							writer.writerow(write_list)
						counter += 1
