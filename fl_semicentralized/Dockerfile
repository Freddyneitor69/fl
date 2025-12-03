FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app
COPY . .
RUN apt-get update && pip install --no-cache-dir -r requirements.txt && pip install --upgrade protobuf

#ENTRYPOINT ["bash", "./entrypoint.sh"]
CMD [ "python", "main.py" ]