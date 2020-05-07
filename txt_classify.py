import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
      
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pd.set_option('max_columns', 50)


'''
NLP上手教程实现

reference：https://github.com/FudanNLP/nlp-beginner

任务一：基于机器学习的文本分类
实现基于logistic/softmax regression的文本分类

TIPs:
    
    Pipeline
    




'''



if __name__=='__main__':
    
    os.chdir('/home/jzy/Desktop/NLP')
    init_data=pd.read_csv(r'./data/sentiment-analysis-on-movie-reviews/train.tsv', sep='\t')
    test_data=pd.read_csv(r'./data/sentiment-analysis-on-movie-reviews/test.tsv', sep='\t')
    
    # EDA
    print(init_data.head())
    print(init_data.info()) 
    
    sentiment_percentage=init_data['Sentiment'].value_counts()/init_data['Sentiment'].count()
    print(sentiment_percentage)
    
    X_train = init_data['Phrase']
    y_train = init_data['Sentiment']
    
    classifier=TfidfVectorizer()
    text_clf = classifier.fit(X_train,y_train)
    sparse_result = text_clf.transform(X_train)      # 得到tf-idf矩阵，稀疏矩阵表示
    print(sparse_result)
    print(text_clf.vocabulary_)                      # 词语与列的对应关系

    
#        
#    X_test = test_data['Phrase']
#    phraseIds = test_data['PhraseId']
#    predicted = text_clf.predict(X_test)
#    pred = [[index+156061,x] for index,x in enumerate(predicted)]
#
#
#    
#    
#    
    
    
    
    
    
    
    '''

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LogisticRegression()),])
    
    
    
    text_clf = text_clf.fit(X_train,y_train)
    
    
    
    
    
    X_test = df.head()['Phrase']
    predicted = text_clf.predict(X_test)
    print np.mean(predicted == df.head()['Sentiment'])
    for phrase, sentiment in zip(X_test, predicted):
        print('%r => %s' % (phrase, sentiment))
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # 提取特征，一般选择 bag-of-words, tf-idf,n-gran

    # 1. bag of words
    
    #corpus=
    
    

    # 2.tf-idf

    






    pass


