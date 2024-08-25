# Sentiment-Analyis

##Sentiment Analysis on Movie Reviews

Overview
This project performs sentiment analysis on movie reviews using the IMDb dataset. The goal is to classify movie reviews as positive or negative based on the textual content. We use the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique for feature extraction and Naive Bayes as the machine learning classifier.

Project Structure
bash
Copy code
Sentiment-Analysis/
│
├── IMDB Dataset.csv                            # Dataset containing movie reviews
├── Movie sentiment analysis using TF_IDF and Naive Bayes.ipynb   # Jupyter Notebook with analysis code
├── README.md                                   # Project documentation
Dataset
The dataset used in this project is the IMDb movie reviews dataset (IMDB Dataset.csv). It contains a collection of movie reviews, each labeled as either positive or negative. This dataset is widely used for binary sentiment classification tasks.

Columns:
review: The text of the movie review.
sentiment: The sentiment label, either positive or negative.
Methodology
Preprocessing:

The text data is cleaned by removing HTML tags, converting the text to lowercase, and removing stopwords.
Further, punctuation and special characters are stripped to reduce noise.
Feature Extraction:

The TF-IDF technique is used to convert text data into numerical features. This representation helps highlight the importance of words in a review while reducing the effect of common words that appear frequently across all reviews.
Modeling:

We use the Naive Bayes classifier to perform binary classification on the reviews. Naive Bayes is chosen for its simplicity and effectiveness in text classification tasks.
Evaluation:

The model is evaluated using accuracy, precision, recall, and F1-score metrics.
Installation and Requirements
Prerequisites:
Python 3.x
Jupyter Notebook
Libraries:
pandas
numpy
scikit-learn
nltk (for natural language processing)
Installation:
Clone the repository:
git clone https://github.com/shaileshkry/Sentiment-Analysis.git
Install the required libraries:
pip install pandas numpy scikit-learn nltk
Open the Jupyter Notebook:
jupyter notebook "Movie sentiment analysis using TF_IDF and Naive Bayes.ipynb"
Usage
Load the IMDb dataset (IMDB Dataset.csv) into the notebook.
Preprocess the text data by cleaning and normalizing the text.
Convert the cleaned text into TF-IDF features.
Train a Naive Bayes classifier on the transformed data.
Evaluate the model on the test dataset and view the performance metrics.
Results
The model achieves a reasonable level of accuracy in classifying movie reviews as either positive or negative. The final results, including accuracy, precision, recall, and F1-score, are displayed at the end of the notebook. These metrics provide insight into the model’s performance on the test set.

Contributing
If you would like to contribute to this project, feel free to fork the repository and submit pull requests. Contributions are always welcome!

License
This project is open-source and available under the MIT License.

