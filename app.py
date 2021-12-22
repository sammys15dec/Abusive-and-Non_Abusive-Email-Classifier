import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import cleantext
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

st.title('Abusive Email classification')
st.subheader("Natural Language processing")
def user_input_features():
    content = st.text_input("Enter mail")
    return content

df = user_input_features()
st.write(df)

data = pd.read_csv('emails1.csv')
data.dropna(inplace=True)
data['Class']=encoder.fit_transform(data['Class'])
data1 = data.head(20000)

vectorizer = TfidfVectorizer(min_df = 5, max_df = 0.8, sublinear_tf = True, use_idf = True)
X = vectorizer.fit_transform(data1['content'])


classifier = LogisticRegression(solver='liblinear',penalty='l1')
classifier.fit(X, data1['Class'])

df=cleantext.clean(df)
df= vectorizer.transform([df])
prediction = classifier.predict(df)
st.subheader('Predict')
st.write('ABUSIVE' if prediction == 0 else 'NON ABUSIVE')
