FROM python:3.11-slim
WORKDIR /app
COPY Fuel_Efficiency_Intelligence/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY Fuel_Efficiency_Intelligence ./Fuel_Efficiency_Intelligence
WORKDIR /app/Fuel_Efficiency_Intelligence
ENV PORT=8000
EXPOSE 8000
CMD ["python", "app/app.py"]

