FROM quay.io/jupyter/scipy-notebook
LABEL authors="lucas"

# get the requirements and install them
COPY requirements_notebook.txt ./requirements_notebook.txt
RUN pip install -r requirements_notebook.txt

CMD jupyter notebook --ip 0.0.0.0 --port 8081 --no-browser --allow-root /app/plots_results_paper.ipynb