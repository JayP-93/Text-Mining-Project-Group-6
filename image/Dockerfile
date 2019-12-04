#######################
# Choose an OS or runtime environment image that fits the needs of your experiment e.g.
#FROM debian:jessie
#Or:
FROM rappdw/docker-java-python:latest
#######################

#Define input/output directories
VOLUME "/input"
VOLUME "/output/datasets"
VOLUME "/output/tables_and_plots"

#######################
# Customization start #
#######################

ADD requirements.txt /requirements.txt
RUN pip install -r requirements.txt
VOLUME "/code"

#######################
# Customization end   #
#######################

#Add and set entrypoint
ADD run.sh /run.sh
RUN chmod u+x /run.sh
ENTRYPOINT /run.sh