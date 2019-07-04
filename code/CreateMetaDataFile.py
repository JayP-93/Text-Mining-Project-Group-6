"""
Create a metadata file (for each language) for MERLIN corpus data, listing all other proficiency labels.
"""

import os

dirpath = "/Users/macbookg/Downloads/meta_ltext"  # path to original corpora folder
outputdirpath = "/Users/macbookg//Downloads/Renamed/"  # path to folder with output files

inputdirs = ["czech", ]  # names of input folders
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
