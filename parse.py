
import os
import sys
import shutil

#type in the path of the .cha files in terminal: python3 parse.py path/to/dir
#results are saved in dir_text

directory = str(sys.argv[1])
#print(os.listdir(directory))
if os.path.exists(directory+"_txt"):
    shutil.rmtree(directory+"_txt")
os.makedirs(directory+"_txt")

for filename in os.listdir(directory):
	if filename != ".DS_Store":
		with open(os.path.join(directory, filename)) as reader:
			fil = open(os.path.join(directory+"_txt", filename[:-3]+"txt"), 'a')
			start = False 
			print(filename)
			for r in reader:
				if r.startswith('*') or start:
					if r.startswith('%') or r.startswith('@'):
						start = False
					else:
						fil.write(r)
						start = True



