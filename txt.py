# Txt processing ( removing numbers, punctuations, white spaces and converting lowercase
import re
import string
for i in range(df_train["text"].size):
    df_train["text"][i]=df_train["text"][i].lower()
    df_train["text"][i]=re.sub(r'\d+', "", df_train["text"][i])
    
for i in range(df_train["text"].size):
    result=df_train["text"][i].translate(str.maketrans('','', string.punctuation))
    df_train["text"][i]=result
    df_train["text"][i]=df_train["text"][i].strip()

  ##TF-ID transform  
from sklearn.feature_extraction.text import TfidfTransformer
def loadData(X_train, X_test,MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    print("tf-idf with",str(np.array(X_train).shape[1]),"features")
    return (X_train,X_test)
