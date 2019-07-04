"""
Create a new dataset with new file names, and removed metadata.
The results (renamed corpora) are in the Datasets/ folder in this repo
Original corpora for all languages can be downloaded from MERLIN website
http://www.merlin-platform.eu/C_data.php
"""

import os

dirpath = "/Users/sowmya/Research/CrossLing-Scoring/Corpora/"  # path to original corpora folder
outputdirpath = "/Users/sowmya/Research/CrossLing-Scoring/Corpora/Renamed/"  # path to folder with output files
files = os.listdir(dirpath)

inputdirs = ["CZ_ltext_txt", "DE_ltext_txt", "IT_ltext_txt"]  # names of input folders
outputdirs = ["CZ", "DE", "IT"]  # names of output folders

for i in range(0, len(inputdirs)):
    files = os.listdir(os.path.join(dirpath, inputdirs[i]))  # original files

    new_folder = os.path.join(outputdirpath, outputdirs[i])  # folder with output files
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for file in files:
        print(file)
        if file.endswith(".txt"):
            content = open(os.path.join(dirpath, inputdirs[i], file), "r").read()
            cefr = content.split("Learner text:")[0].split("Overall CEFR rating: ")[1].split("\n")[0]  # language level
            newname = file.replace(".txt", "") + "_" + outputdirs[i] + "_" + cefr + ".txt"  # name of new file
            fh = open(os.path.join(outputdirpath, outputdirs[i], newname), "w")  # new file
            text = content.split("Learner text:")[1].strip()  # raw text
            fh.write(text)  # write to new file
            fh.close()
            print("wrote: ", os.path.join(outputdirpath, outputdirs[i], newname))
