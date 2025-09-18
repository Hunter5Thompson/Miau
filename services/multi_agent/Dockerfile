# Lightweight container for the multi-agent research workflow
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY Requirments_Agents.txt ./Requirments_Agents.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r Requirments_Agents.txt

COPY services /app/services

ENTRYPOINT ["python", "-m", "services.multi_agent"]
CMD []
