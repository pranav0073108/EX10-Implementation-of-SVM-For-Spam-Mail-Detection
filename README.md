# EX 10 Implementation of SVM For Spam Mail Detection
## DATE:
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect and clean the email dataset: Emails should be labeled as "spam" or "not spam."
2. Split the dataset into training and test sets: Typically, this is done with a ratio of 80% training data and 20% test data.
3. Choose a kernel function: SVM can use various kernel functions
4. Make predictions: Use the trained SVM model to predict whether new emails are spam or not based on the test data.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection.
Developed by: pranav k
RegisterNumber:  2305001026
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score

df=pd.read_csv('/content/spamEX10.csv',encoding='ISO-8859-1')
df.head()


vectorizer=CountVectorizer()
X=vectorizer.fit_transform(df['v2'])
y=df['v1']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


model=svm.SVC(kernel='linear')
model.fit(X_train,y_train)


predictions=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,predictions))
print("Classification Report:")
print(classification_report(y_test,predictions))


def predict_message(message):
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)
    return prediction[0]


new_message="Free prixze money winner"
result=predict_message(new_message)
print(f"The message: '{new_message}' is classified as: {result}")
```

## Output:
![image](https://github.com/user-attachments/assets/4d060d2a-c682-4b93-affe-34506e8b9dc5)

![image](https://github.com/user-attachments/assets/7a7a43d5-e1d7-4794-82a2-2736e82ee3c7)

![image](https://github.com/user-attachments/assets/b2e0d172-9537-411d-b446-18158ab54fb0)

![image](https://github.com/user-attachments/assets/3699c82f-0e95-468c-8778-26d37a4fe361)






## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
