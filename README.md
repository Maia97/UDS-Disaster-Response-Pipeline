# UDS-Disaster-Response-Pipeline

## Description

This Project is part of the Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweets and messages from real-life disaster events. The aim of the project is to build a Natural Language Processing (NLP) model to categorize messages on a real-time basis.

The project consists of three main components: **ETL pipeline, ML pipeline, and Flask web app**. 

In the ETL pipeline, we will load and merge the message and category datasets, clean the data, and store it in a SQLite database. In the ML pipeline, we will build a text processing and machine learning pipeline that will train and tune a model using GridSearchCV. Finally, we will develop a Flask web app that will allow the user to input a new message and get classification results in several categories. The web app will also display data visualizations using Plotly.

## Program Structure
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- ETL Pipeline Preparation.ipynb # data processing step by step
- ML Pipeline Preparation.ipynb # model development step by step

- requirements.txt # required Python packages
- .gitattributes # large file tracking
- .gitignore # ignored files

- README.md
```

## Environment
* Python 3.7
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Model Loading and Saving Library: Pickle
* Web App and Data Visualization: Flask, Plotly

Install Python packages with `pip` and `requirements.txt`
```
$ pip install -r requirements.txt
```

## Executing Program
1. (Optional) Process the database, train and save model.

    - Clean and store data in SQLite database: 
        ```
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db
        ```
    - Loads data, train and save ML classifier as a pickle file: 
        ```
        python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl
        ```

2. Run the web app under the app's directory:
    ```
    app $ python run.py
    ```

3. Open web page at http://0.0.0.0:3000/



## Web App Preview

The main page shows statistic graphs about the training dataset, provided by Figure Eight. We can input a message and click **Classify Message** button to test the model claasifying result.

<img width="563" alt="image" src="https://user-images.githubusercontent.com/65010631/225126153-b24d3038-c0ce-406d-ad07-4dae0bfff13a.png">





