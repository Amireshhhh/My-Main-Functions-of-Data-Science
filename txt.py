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
