FROM python:3.7
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./src .

CMD ["streamlit", "run", "simple_image_clf.py"]