"""
Renaiming of files in new EN corpora (analog to CreateDataset.py
"""

import os

dirpath = "../UAE"  # path to original corpora folder
outputdirpath = "../Datasets/"  # path to folder with output files
outputdir = 'EN'

folders = os.listdir(dirpath)

for i in range(0, len(folders)):
    path_to_folder = os.path.join(dirpath,folders[i])
    if os.path.isdir(path_to_folder):
        files = os.listdir(path_to_folder)  # original files in folder
        for file in files:
            # print(file)
            if file.endswith(".txt"):
                content = open(os.path.join(path_to_folder, file), "r").read()
                cefr = file[-8:-6]
                newname = file.replace(".txt", "").replace("W_UAE_", "")[
                          :-5] + "_" + outputdir + "_" + cefr + ".txt"  # name of new file
                print(newname)
                fh = open(os.path.join(outputdirpath, outputdir, newname), "w")  # new file
                fh.write(content)  # write to new file
                fh.close()
