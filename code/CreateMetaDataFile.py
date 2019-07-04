"""
Create a metadata file (for each language) for MERLIN corpus data, listing all other proficiency labels.
The results (metadata files) are in the Datasets/ folder in this repo
"""

import os

dirpath = "/Users/sowmya/Research/CrossLing-Scoring/Corpora/"  # path to original corpora folder
outputdirpath = "/Users/sowmya/Research/CrossLing-Scoring/Corpora/"  # path to folder with output files

inputdirs = ["CZ_ltext_txt", "DE_ltext_txt", "IT_ltext_txt"]  # names of input folders
outputnames = ["CZMetadata.txt", "DEMetadata.txt", "ITMetadata.txt"]  # names of output files

for i in range(0, len(inputdirs)):

    new_folder = os.path.join(outputdirpath)  # folder with output files
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    outpath = os.path.join(new_folder, outputnames[i])  # path to output file
    files = os.listdir(os.path.join(dirpath, inputdirs[i]))  # original files
    fw = open(outpath, "w")  # output file
    for file in files:
        if file.endswith(".txt"):
            content = open(os.path.join(dirpath, inputdirs[i], file), "r").read()
            ratings = content.split("Rating:")[1].split("\n\n")[0].strip().split("\n")  # get all ratings in a list
            # join the list contents to a string, and remove white spaces and slashes in descriptions.
            fw.write(file.replace(".txt", "") + "," + ",".join(ratings).replace(" ", "").replace("/", ""))
            fw.write("\n")
    fw.close()
    print("Wrote to: ", outpath)
