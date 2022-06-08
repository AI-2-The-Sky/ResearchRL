# from base image debian
FROM python:3.8-bullseye

COPY . /
WORKDIR /

WORKDIR /ExplosiveAI
RUN ls
RUN chmod +x simulator/build/bomber.x86_64
RUN python3 -m pip install build
RUN python3 -m build
RUN python3 -m pip install dist/bomberman-0.1.0-py3-none-any.whl


CMD ["python3", "fight.py"]
