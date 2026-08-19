FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY lambda_func.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_func.lambda_handler"]