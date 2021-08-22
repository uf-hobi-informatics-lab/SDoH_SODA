FROM python:3.8

# Set the working directory to /deepdeid
WORKDIR /sdoh_cor

# Copy the current directory contents into the container at /deepdeid
#COPY ..

# check file directory
#RUN ls -l
#RUN du -h

# Install any needed packages specified in requirements.txt
# RUN pip install -r requirements.txt && python -m spacy download en_core_web_lg
#RUN pip install -r requirements.txt && python -m spacy download en_core_web_sm

RUN chmod +x ./sdoh_cor/run_pred.sh

# Define the entrypoint for the container as end2end_docker.sh
ENTRYPOINT ["./sdoh_cor/run_pred.sh"]

# Run end2end.sh when the container launches
CMD ["./data/test_set_update/",'test', "-1"]
