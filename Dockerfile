FROM jupyter/base-notebook:x86_64-python-3.11.6
LABEL authors="lucas"

# get the requirements and install them
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

ENTRYPOINT jupyter notebook --ip 0.0.0.0 --port 8081 --no-browser --allow-root /app/plots_results_paper.ipynb