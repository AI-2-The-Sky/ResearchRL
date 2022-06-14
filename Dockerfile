# from base image debian
FROM bomberman-explosiveai

COPY stable_baseline_agent /ResearchRL/stable_baseline_agent

WORKDIR /ResearchRL
RUN rm -rf ExplosiveAI

CMD ["python3", "stable_baseline_agent/agent/train.py"]
