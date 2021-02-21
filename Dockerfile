FROM python:3.8
COPY . /app
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 80
RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml

RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'\
    'libgl1-mesa-glx'\
    'libgl1-mesa-dev'  -y

WORKDIR /app
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]