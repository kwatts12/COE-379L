#Image: kmw4568/ml-satellite-api

FROM python:3.11

RUN pip install tensorflow==2.15
RUN pip install Flask==3.1.0
RUN pip install Pillow

COPY models /models
COPY inferenceServer.py /inferenceServer.py


CMD ["python", "inferenceServer.py"]