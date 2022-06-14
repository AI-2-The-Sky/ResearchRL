# from base image debian
FROM python:3.8-bullseye

WORKDIR /ResearchRL/ExplosiveAI
RUN chmod +x simulator/build/bomber.x86_64
RUN python3 -m pip install build
RUN python3 -m build
RUN python3 -m pip install dist/bomberman-0.1.0-py3-none-any.whl

WORKDIR /ResearchRL

CMD ["python3", "stable_baseline_agent/agent/train.py"]
