'''A company is looking to develop a text classification model to categorize customer reviews into positive or negative sentiments.They want to use Naive Bayes classification to automatically analyze
and classify the reviews they receive, aiming to understand customer satisfaction levels and sentiments. The company desires a model that can accurately predict whether a customer review expresses a positive or negative sentiment.'''
# this model is used when we have to deal with text

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer # It will convert text into matrix / tokenize the counting
from sklearn.naive_bayes import MultinomialNB

# step-2
reviews =["The product is excellent and works perfectly","the product is not good, very disappointing","terrible product and waste of money","i love this product and it is amazing"]
sentiments = np.array([1,0,0,1]) # 1:positive , 0:Negative

# step-3 We will train and test ....CountVectorizer has inbuilt features for testing and training
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(reviews)

# step-4 we have to do training and testing on whole data not on some precentage
classifier=MultinomialNB()
classifier.fit(X,sentiments) 

# step-5
def classify_new_review(review):
    review_vectorized=vectorizer.transform([review])
    prediction=classifier.predict(review_vectorized)
    if prediction[0]==1:
        return "Postive statement"
    else:
        return "Negative statement"

# step-6 user input
user_review=input("Enter your review: ")
result=classify_new_review(user_review)
print(f"The review {user_review} is classified as {result}")


