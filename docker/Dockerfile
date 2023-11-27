# we use the SageMaker pre-built SKLearn image as the base image
# for this example, we use the us-east-1 region. If you have a different region
# please update the image uri below accordingly
FROM 683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3

RUN python3 -m pip install awswrangler

ENV PYTHONUNBUFFERED=TRUE

ADD ./src/model_quality_monitoring.py /

ENTRYPOINT ["python3", "/model_quality_monitoring.py"]
