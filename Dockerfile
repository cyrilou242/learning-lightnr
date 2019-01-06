FROM python:3.6.7-stretch
MAINTAINER Cyril de Catheu @AB Tasty
RUN apt-get update --yes
RUN pip install --upgrade setuptools
RUN pip install pandas
RUN pip install nltk
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('punkt')"
RUN pip --no-cache-dir install spacy
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_md
RUN pip install sumy
RUN pip install beautifulsoup4
RUN pip install flask
RUN pip install gunicorn
RUN pip install allennlp
RUN pip install scikit-learn==0.19.0
ADD app /app
WORKDIR app/
CMD gunicorn -w 1 app:app --limit-request-line 20000 -b '0.0.0.0:5000' --timeout 100

