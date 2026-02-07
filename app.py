import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
app = Flask(__name__)
@app.route('/',methods=['Get','Post'])
def index():
     return render_template('Index.html')
@app.route('/home',methods=['Get','Post'])
def home():
    output=""
    msg=""
    return render_template('Home.html', prediction=output, msg=msg)
def getsentiment(text):
    sentiment=""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    from textblob import TextBlob
    t = "i have finished my work so i am so happy"
    sentiment = TextBlob(t).sentiment
    print(sentiment)

    import pandas as pd

    df = pd.read_csv('IMDB Dataset.csv', nrows=5000)
    df.head(10)

    df

    df.tail()

    df.describe()

    df.info()

    df.shape

    df.fillna(0, inplace=True)

    df.info()

    df['review'][0]

    df['sentiment'].replace({'positive':1, 'negative':0}, inplace=True)

    df.head()

    import re
    print(df.iloc[0].review)

    clean=re.compile('<.*?>')
    re.sub(clean,'',df.iloc[0].review)

    def clean_html(text):
        clean=re.compile('<.*?>')
        return re.sub(clean,'',text)

    df['review']=df['review'].apply(clean_html)

    df.head()

    def convert_lower(text):
        return text.lower()

    df['review']=df['review'].apply(convert_lower)

    df.head()

    def remove_special(text):
        x=''
        for i in text:
            if i.isalnum():
                x=x+i
            else:
                x=x+' '
        return x

    remove_special('Th%e @ classic use of the word. it is called oz as that is the nickname given to the oswald maximum security state')

    df['review']=df['review'].apply(remove_special)

    df.head()

    import nltk

    #nltk.download('stopwords')

    from nltk.corpus import stopwords
    stopwords.words('english')

    def remove_stopwords(text):
        x=[]
        for i in text.split():
            if i not in stopwords.words('english'):
                x.append(i)
        y=x[:]
        x.clear()
        return y

    df['review']=df['review'].apply(remove_stopwords)

    df.head()

    y=[]
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    def stem_words(text):
        for i in text:
            y.append(ps.stem(i))
        z=y[:]
        y.clear()
        return z

    stem_words(['I','loved','loving','went', 'mentioned'])

    df['review']=df['review'].apply(stem_words)

    df.head()

    def join_back(list_input):
        return " ".join(list_input)

    df['review']=df['review'].apply(join_back)

    df.head()

    from sklearn.feature_extraction.text import CountVectorizer
    cv=CountVectorizer(max_features=150)

    X=cv.fit_transform(df['review']).toarray()

    X[1:5,:]

    X.shape

    X[0].mean()

    y=df.iloc[:,-1].values

    y

    from sklearn.model_selection import train_test_split
    X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.2)

    y_train.shape

    X_train.shape

    X_test.shape

    from sklearn.naive_bayes import GaussianNB

    clf1=GaussianNB()

    clf1.fit(X_train,y_train)

    y_pred1=clf1.predict(X_test)

    from sklearn.metrics import accuracy_score

    print("Gaussian",accuracy_score(y_test,y_pred1))
    new_review = text

    # Redefine stem_words to use a local variable instead of the global 'y'
    # The original stem_words function was defined in cell OG_C0k0oCEQ1 and used a global 'y'.
    # This caused a conflict when 'y' was later reassigned to df.iloc[:,-1].values (a numpy array).
    # To fix this, we should make the list for collecting stemmed words local to the function.

    def stem_words(text):
        local_stemmed_words = [] # Use a local variable here
        for i in text:
            local_stemmed_words.append(ps.stem(i))
        return local_stemmed_words

    def preprocess_review(review):
        # Clean HTML
        review = clean_html(review)
        print("clean html", review)
        # Convert to lowercase
        review = convert_lower(review)
        print("convert lower", review)
        # Remove special characters
        review = remove_special(review)
        print("remove_special ", review)
        # Remove stopwords
        review_words = remove_stopwords(review)
        print("remove_stopwords ", review_words)
        # Stem words
        review_stemmed = stem_words(review_words) # This now calls the locally fixed stem_words
        print("stem_words ", review_stemmed)
        # Join back to string
        review = join_back(review_stemmed)
        print("join_back ", review)
        return review

    processed_new_review = preprocess_review(new_review)

    # Vectorize the preprocessed review
    X_new = cv.transform([processed_new_review]).toarray()

    # Predict the sentiment
    prediction = clf1.predict(X_new)
    print(prediction)

    if prediction[0] == 1:
        sentiment="Positive"
        print(f"The review '{new_review}' is predicted as: Positive")
    else:
        sentiment="Negative"
        print(f"The review '{new_review}' is predicted as: Negative")
    return sentiment

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    msg=''
    output=""
    if request.method == 'POST':
        s = request.form['sentiment']
        print(s)
        output = getsentiment(s)
        print(output)
        msg='Sentiment Analysis Result'+output
        return render_template('Index.html', prediction=output, msg=msg)
def getfakenewsprediction(text):
    # -*- coding: utf-8 -*-
    from textblob import TextBlob
    t = "i have finished my work"
    sentiment=TextBlob(t).sentiment
    print(sentiment)

    #1. Import dataset
    #2. Categorical to Numerical (Sentiment Column)
    #3. Preprocessing - Remove HTML
    #4. Lower Case Conversion
    #5. Remove Special Characters
    #6. Stop Word Removal
    #7. Stemming Process
    #8. Joining the list of words
    #9. Feature Extraction using count vectorize
    #10. Train Test Split
    #11. Create Algorithm
    #12. Fit the Training Data
    #13. Predict the testing data
    #14. Performance Evaluation (Accuracy)

    import pandas as pd

    df=pd.read_csv('FakeNewsDataset.csv', nrows=5000)
    df.head(10)

    df.tail()

    df.describe()

    df.info()

    df.shape

    df.fillna(0,inplace=True)

    df.info()

    df['Text'][1]

    df['Sentiment'].replace(True,1,inplace=True)
    df['Sentiment'].replace(False,0,inplace=True)

    df.head()

    import re
    print(df.iloc[2].Text)

    clean=re.compile('<.*?>')
    print(re.sub(clean,'',df.iloc[1].Text))

    def clean_html(text):
        clean=re.compile('<.*?>')
        return re.sub(clean,'',text)

    df['Text']=df['Text'].apply(clean_html)

    df.head()

    def convert_lower(text):
        return text.lower()

    df['Text']=df['Text'].apply(convert_lower)

    df.head()

    def remove_special(text):
        x=''
        for i in text:
            if i.isalnum():
                x=x+i
            else:
                x=x+' '
        return x

    remove_special("Th%e @ classic use of the word. it is called oz as that is the nickname given to the oswald maximum")

    df['Text']=df['Text'].apply(remove_special)

    df.head()

    import nltk
    nltk.download('stopwords')

    from nltk.corpus import stopwords
    stopwords.words('english')

    def remove_stopwords(text):
        x=[]
        for i in text.split():
            if i not in stopwords.words('english'):
                x.append(i)
        y=x[:]
        x.clear()
        return y

    df['Text']=df['Text'].apply(remove_stopwords)

    print (df.head())

    y=[]
    from nltk.stem import PorterStemmer
    ps=PorterStemmer()
    def stem_words(text):
        for i in text:
            y.append(ps.stem(i))
        z=y[:]
        y.clear()
        return z

    stem_words(['I','loved','loving','went','mentioned'])

    df['Text']=df['Text'].apply(stem_words)

    df.head()

    def join_back(list_input):
        return " ".join(list_input)

    df['Text']=df['Text'].apply(join_back)

    df.head()

    from sklearn.feature_extraction.text import CountVectorizer
    cv=CountVectorizer(max_features=150)

    X=cv.fit_transform(df['Text']).toarray()

    X[1:5,:]

    X.shape

    y=df.iloc[:,-1].values

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

    X_train.shape

    y_train.shape

    X_test.shape

    y_test.shape

    from sklearn.naive_bayes import GaussianNB
    clf1=GaussianNB()
    clf1.fit(X_train,y_train)

    y_pred1=clf1.predict(X_test)

    from sklearn.metrics import accuracy_score
    print("Gaussian", accuracy_score(y_test,y_pred1))

    from sklearn.ensemble import RandomForestClassifier

    clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf1.fit(X_train,y_train)

    y_pred1=clf1.predict(X_test)
    from sklearn.metrics import accuracy_score
    print("Random Forest", accuracy_score(y_test,y_pred1))

    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(criterion='entropy', random_state=1)
    model.fit(X_train, y_train)
    y_pred2 = model.predict(X_test)
    from sklearn.metrics import accuracy_score
    print("Decision Tree", accuracy_score(y_test,y_pred2))

    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred3 = model.predict(X_test)
    from sklearn.metrics import accuracy_score
    print("KNN", accuracy_score(y_test,y_pred3))

    from sklearn.svm import SVC
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred4 = model.predict(X_test)
    from sklearn.metrics import accuracy_score
    print("SVC", accuracy_score(y_test,y_pred4))

    new_review = text
    #new_review = '''(Reuters) - Alabama officials on Thursday certified Democrat Doug Jones the winner of the state’s U.S. Senate race, after a state judge denied a challenge by Republican Roy Moore, whose campaign was derailed by accusations of sexual misconduct with teenage girls. Jones won the vacant seat by about 22,000 votes, or 1.6 percentage points, election officials said. That made him the first Democrat in a quarter of a century to win a Senate seat in Alabama.  The seat was previously held by Republican Jeff Sessions, who was tapped by U.S. President Donald Trump as attorney general. A state canvassing board composed of Alabama Secretary of State John Merrill, Governor Kay Ivey and Attorney General Steve Marshall certified the election results. Seating Jones will narrow the Republican majority in the Senate to 51 of 100 seats. In a statement, Jones called his victory “a new chapter” and pledged to work with both parties. Moore declined to concede defeat even after Trump urged him to do so. He stood by claims of a fraudulent election in a statement released after the certification and said he had no regrets, media outlets reported. An Alabama judge denied Moore’s request to block certification of the results of the Dec. 12 election in a decision shortly before the canvassing board met. Moore’s challenge alleged there had been potential voter fraud that denied him a chance of victory. His filing on Wednesday in the Montgomery Circuit Court sought to halt the meeting scheduled to ratify Jones’ win on Thursday. Moore could ask for a recount, in addition to possible other court challenges, Merrill said in an interview with Fox News Channel. He would have to complete paperwork “within a timed period” and show he has the money for a challenge, Merrill said. “We’ve not been notified yet of their intention to do that,” Merrill said. Regarding the claim of voter fraud, Merrill told CNN that more than 100 cases had been reported. “We’ve adjudicated more than 60 of those. We will continue to do that,” he said.  Republican lawmakers in Washington had distanced themselves from Moore and called for him to drop out of the race after several women accused him of sexual assault or misconduct dating back to when they were teenagers and he was in his early 30s.  Moore has denied wrongdoing and Reuters has not been able to independently verify the allegations.'''
    # Redefine stem_words to use a local variable instead of the global 'y'
    # The original stem_words function was defined in cell OG_C0k0oCEQ1 and used a global 'y'.
    # This caused a conflict when 'y' was later reassigned to df.iloc[:,-1].values (a numpy array).
    # To fix this, we should make the list for collecting stemmed words local to the function.

    def stem_words(text):
        local_stemmed_words = [] # Use a local variable here
        for i in text:
            local_stemmed_words.append(ps.stem(i))
        return local_stemmed_words

    def preprocess_review(review):
        # Clean HTML
        review = clean_html(review)
        print("clean html", review)
        # Convert to lowercase
        review = convert_lower(review)
        print("convert lower", review)
        # Remove special characters
        review = remove_special(review)
        print("remove_special ", review)
        # Remove stopwords
        review_words = remove_stopwords(review)
        print("remove_stopwords ", review_words)
        # Stem words
        review_stemmed = stem_words(review_words) # This now calls the locally fixed stem_words
        print("stem_words ", review_stemmed)
        # Join back to string
        review = join_back(review_stemmed)
        print("join_back ", review)
        return review

    processed_new_review = preprocess_review(new_review)

    # Vectorize the preprocessed review
    X_new = cv.transform([processed_new_review]).toarray()

    # Predict the sentiment
    prediction = clf1.predict(X_new)
    print(prediction)
    sentiment=""
    if prediction[0] == 1:
        print(f"The review '{new_review}' is predicted as:")
        print("Real News")
        sentiment="Real News"
    else:
        print(f"The review '{new_review}' is predicted as:")
        print("Fake News")
        sentiment="Fake News"
    return sentiment
@app.route('/fakenewsprediction', methods=['GET', 'POST'])
def fakenewsprediction():
    msg=''
    output=""
    if request.method == 'POST':
        s = request.form['sentiment']
        print(s)
        output = getfakenewsprediction(s)
        print(output)
        msg='Sentiment Analysis Result: '+output
        return render_template('Home.html', prediction=output, msg=msg)
if __name__ == '__main__':
    app.run(port=5000,debug=True)