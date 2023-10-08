FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN yum install -y gcc git mesa-libGL
RUN pip install -r requirements.txt

# COPY app.py user.py sticker.py image.py dynamodb.py emojis.py ${LAMBDA_TASK_ROOT}
COPY weights/u2net.onnx ${LAMBDA_TASK_ROOT}/.u2net/
COPY lambda_function.py image.py ${LAMBDA_TASK_ROOT}
ENV NUMBA_CACHE_DIR=/tmp

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.lambda_handler" ]
