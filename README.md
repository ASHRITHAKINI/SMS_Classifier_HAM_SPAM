# SMS_Classifier_HAM_SPAM

Overview:
  The SMS Classifier: SPAM and HAM is a natural language processing (NLP) project aimed at building a machine learning model to classify SMS messages into two categories: "SPAM" and "HAM" (non-spam). The project leverages popular Python libraries such as Pandas, NumPy, Matplotlib, Seaborn, NLTK, Scikit-learn, and WordCloud for data manipulation, visualization, text processing, and machine learning tasks.

Key Features:
  Data Collection: The project starts with the collection of a labeled dataset containing SMS messages tagged as either "SPAM" or "HAM."
  Exploratory Data Analysis (EDA): Initial exploration and visualization of the dataset are performed to understand its distribution and characteristics.
  Data Preprocessing: Text preprocessing techniques such as tokenization, converting text to lowercase, and removing stopwords are applied to clean the SMS messages.
Feature Engineering: The TfidfVectorizer from Scikit-learn is used to convert the preprocessed text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
  Model Training: The Multinomial Naive Bayes classifier from Scikit-learn is trained on the TF-IDF transformed features to build a predictive model for classifying SMS messages as spam or non-spam.
  Model Evaluation: The trained model's performance is evaluated using classification metrics such as accuracy, precision, recall, and F1-score. Confusion matrix visualization is also generated to analyze the model's predictive performance.
  Result Presentation: Classification report and accuracy score are presented visually to provide a summary of the model's performance on the test data.
  
Dependencies:
  Python 3.x
  pandas
  numpy
  matplotlib
  seaborn
  nltk
  scikit-learn
  wordcloud
  
Dataset:
  The project utilizes a labeled dataset containing SMS messages, where each message is tagged as either "SPAM" or "HAM." The dataset is provided in a CSV file format named 'train.csv.'

Usage:
  1.Clone the repository to your local machine.
  2.Install the required dependencies listed in the 'requirements.txt' file.
  3.Run the provided Python script 'sms_classifier.py' to execute the SMS Classifier.
  4.Follow the instructions provided in the script to load the dataset, preprocess the data, train the model, and evaluate its performance.
  5.View the generated visualizations and classification report to analyze the model's performance.

Acknowledgements:
  The project was inspired by the need for efficient spam detection mechanisms in communication systems.
Special thanks to the developers and contributors of the open-source libraries and datasets used in this project.
