# from base image debian
FROM bomberman-explosiveai

COPY . /ResearchRL

WORKDIR /ResearchRL
RUN rm -rf ExplosiveAI

CMD ["python3", "stable_baseline_agent/agent/train.py"]
