This repository participates to [Reprolang 2020](https://www.clarin.eu/event/2020/reprolang-2020). It contains the source code files, the execution environment and the data used for experiments containing texts in 4 langauges: Italian, German, Czech and English.

The build-script, image directories are created here according to the instructions for the docker provided by Reprolang.

The container for the docker exceeds 1 GB, so it can not be uploaded on gitlab.

In order to build the docker locally, go to the project location and execute

`./build.sh --build --local`

In order to test the built docker, execute:

```
    docker run \
    -ti --rm --name=test \
    -v ${PWD}/code:/code \
    -v ${PWD}/input:/input \
    -v ${PWD}/output/datasets:/output/datasets \
    -v ${PWD}/output/tables_and_plots:/output/tables_and_plots \
    <image-name>:<image-id>
```

The project requires a input and output folder in the root.

The execution with the English language is taking a lot of time. If English is excluded, please delete the following lines:

/image/run.sh
`
python /code/monolingual_cv.py /input/EN > /output/datasets/monolingual_word_emb_en.txt`

/code/IdeaPOC.py
`
endirpath = "../input/EN-Parsed"
`
and all the lines using english for single language, cross-language. Delete English from mega_multilingual_model too.



The dataset is in datasets.tar.gz . It should be extracted such that the input would contain directly the language folders.

The md5 checksum for datasets.tar.gz is `52bd58d77e870c4db131d94d4f1f4146`.

The tag to look for is `docker-version`



Original project GitHub repository: https://github.com/nishkalavallabhi/UniversalCEFRScoring


Original README:
----------------------------------------------------------------------------------------------------------
This repository contains the code and some of the data and result files for the experiments described in the following paper:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1218081.svg)](https://doi.org/10.5281/zenodo.1218081)

> Experiments with Universal CEFR Classification  
> Authors: Sowmya Vajjala and Taraka Rama  
> (to appear) In Proceedings of The 13th Workshop on Innovative Use of NLP for Building Educational Applications

For enquiries, contact one of the authors, at:
sowmya@iastate.edu, tarakark@ifi.uio.no

About this github repo's folders:  
  
- **Datasets/:**  
  * This contains folders with learner corpora for three languages(German-DE, Italian-IT, Czech-CZ), from [MERLIN project](http://www.merlin-platform.eu/), and their dependency parsed versions (DE-Parsed, IT-Parsed, CZ-Parsed) using [UDPipe](http://ufal.mff.cuni.cz/udpipe).
  * The [original MERLIN corpus](http://www.merlin-platform.eu/C_data.php) is available under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/), which allows us to: "copy and redistribute the material in any medium or format".
  * The original corpus files have been renamed for our purposes, and all metadata showing information about the learner background, different dimensions of proficiency etc have been removed. All .Metadata.txt files in this folder contain information about other CEFR proficiency dimensions for the learner writings, which we did not use in this paper.
  * RemovedFiles/ folder in this folder contains files that we did not use for our analysis, either because they are unrated, or because that proficiency level has too few examples (Read the paper for details).

- **code/:**
  * CreateDataset.py: takes the original MERLIN corpus and creates a new version by removing metadata inside files, and renaming them.
  * CreateMetaDataFile.py: takes the original MERLIN corpus files, and creates a file with filenames and information about
different proficiency dimensions (resulting files for all languages are seen in Datasets/)
  * ErrorStats.py: uses [LanguageTool](https://languagetool.org/) to extract spelling/grammar error information for DE and IT corpora.
  * ExtractErrorFeatures.py: extracts information about individual error rules from LanguageTool, and stores them as numpy arrays - this is not used in this paper.
  * IdeaPOC.py: contains the bulk of the code - basically, all experiments without neural networks are seen here. 
  * bulklangparse.sh: Script to parse all files for a given language using its UDPipe model.  
  * monolingual_cv.py: mono-lingual neural network training using keras, with tensorflow backend.
  * multi_lingual.py: multi-lingual, multi-task learning (learning the language, and learning its CEFR)
  * multi_lingual_no_langfeat.py: multi-lingual, without language identification.  
  (For more details, read the paper, and see the code!)

- **features/:**
  * This directory contains some analysis of the data using [LanguageTool](https://languagetool.org/) to understand the different error types in different languages, as identified by the tool. Only the files ending with errorcats.txt have been used in this paper.

- **results/:**
  * This folder contains some the result files generated while doing the experiments, primarily to keep a record. It is useful for replication and comparison purposes in future.

- **README.md**: This file  

- **bea-naacl2018-supplementarymaterial.zip**: Submitted/pre-peer-reviewed version of this folder for BEA. The current version primarily has more documentation, and added code for baseline classifiers.

- **notesonfeatures.txt**: Just some initial notes made about what to implement.



