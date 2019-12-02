#######################
# Choose an OS or runtime environment image that fits the needs of your experiment e.g.
#FROM debian:jessie
#Or:
FROM continuumio/miniconda3
#######################

#Define input/output directories
VOLUME "/input"
VOLUME "/output/datasets"
VOLUME "/output/tables_and_plots"

#Add and set entrypoint
ADD run.sh /run.sh
RUN chmod u+x /run.sh
ENTRYPOINT /run.sh

#######################
# Customization start #
#######################

RUN conda env create -f env.yml \
&& conda activate tmp

#######################
# Customization end   #
#######################
