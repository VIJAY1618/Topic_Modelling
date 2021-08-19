
"""Importing necessary libraries"""
from sklearn.datasets import fetch_20newsgroups#inbuilt dataset from the sklearn library
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem  import WordNetLemmatizer
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')
import gensim #gensim for implementing LDA model
from gensim import corpora 
from gensim.models import CoherenceModel 

class LDA:
  
  """Method that load the dataset """
  def load_data(self):
    print("Latent Dirichlet Allocation (LDA) with different Topics ")
    train_dataset=fetch_20newsgroups(subset='train',remove=('headers','footers','qutes'))#loading the inbuiltdataset
    data=train_dataset.data
   
    df_news=pd.DataFrame(data)#creating dataframe
    print("length of the dataset",len(data))
    news_df=df_news.sample(1000)#loading 1000 random sample from the dataset
    news_df.reset_index(inplace=True,drop=True)
    return news_df

  """Method to  preprocess the text"""

  def data_preprocessing(self,news_df):
      data=news_df
      df_news=data.rename(columns={0:'News'})#column  renaming
      df_news['News']=df_news.News.apply(lambda x:str(x).lower())#converting the text into lowercase
      df_news['News']=df_news.News.str.replace('[^A-Za-z]+',' ')#removing number and non-alphabetic character
      lemma=WordNetLemmatizer()#creating the object WordNetLemmatizer
      nltk.download('wordnet')
      nltk.download('stopwords')
      nltk.download('punkt')
      words_tokenize=df_news['News'].apply(word_tokenize)#tokenizing the document
      stop_words=stopwords.words('english')
      words_list=[] #list to store the word that are not in stop_words
      for outer_element in words_tokenize:
        for inner_element in outer_element:
          if inner_element not in stop_words:
            words_list.append(inner_element)#storing the word that is not in stop_words
      words=[ele for ele in words_list if len(ele)>3]#removing the word which length is less than 3
      lemmatize_word=[lemma.lemmatize(word) for word in words]#text normalization
      word_split=[s.split() for s in lemmatize_word]#converting each word into list 
      corpus=corpora.Dictionary(word_split)#creating  corpus
      dtm=[corpus.doc2bow(i) for i in word_split]#creating document term matrix 
      return  dtm,corpus,word_split

  """LDA model with 15 topics"""
  def  lda_model1(self,dtm,corpus,word_split):
    dtm=dtm
    corpus=corpus
    word_split=word_split
    lda=gensim.models.ldamodel.LdaModel
    lmodel=lda(dtm,num_topics=15,id2word=corpus,passes=8)#training the LDA model
    coherence_model=CoherenceModel(model=lmodel,texts=word_split,dictionary=corpus,coherence='c_v')
    coherence_score=coherence_model.get_coherence()
    print("Coherence score with 15 Topics ",coherence_score)

  """LDA model with 10 topics"""
  def  lda_model2(self,dtm,corpus,word_split):
    dtm=dtm
    corpus=corpus
    word_split=word_split
    lda2=gensim.models.ldamodel.LdaModel
    lmodel2=lda2(dtm,num_topics=10,id2word=corpus,passes=8)
    coherence_model=CoherenceModel(model=lmodel2,texts=word_split,dictionary=corpus,coherence='c_v')
    coherence_Score=coherence_model.get_coherence()
    print("Coherence score with 10 topics",coherence_Score)

    """LDA model with 5 topics"""
  def  lda_model3(self,dtm,corpus,word_split):
    dtm=dtm
    corpus=corpus
    word_split=word_split
    lda3=gensim.models.ldamodel.LdaModel
    ldamodel3=lda3(dtm,num_topics=5,id2word=corpus,passes=8)
    coherence_model=CoherenceModel(model=ldamodel3,texts=word_split,dictionary=corpus,coherence='c_v')
    coherence_Score=coherence_model.get_coherence()
    print("Coherence score with 5  topics",coherence_Score)
    for topic_id in range(ldamodel3.num_topics):
      topics=ldamodel3.show_topic(topic_id)
      word_topic=[word for word,_ in topics]
      print(f'Topic {topic_id}:{word_topic}')#words in each topics
    return ldamodel3

  """Method to predict the new instance with best  Coherence Socre  Model"""
  def predict(self,lda3):
    lda3=lda3
    text="Computer is a an eletronic devices that take input,process it and generate the user desired output"
    words=word_tokenize(text)
    word=[]
    stop_words=stopwords.words('english')
    for i in words:
      if  i not in stop_words:
        word.append(i)
    lower_case_word=[lowercase.lower() for lowercase in word]
    lemma=WordNetLemmatizer()
    word_len=[ele for ele in lower_case_word if len(ele)>3]
    word_lemmatize=[lemma.lemmatize(i) for i in word_len]
    word_splits=[g.split() for g in word_lemmatize]
    corpus=corpora.Dictionary(word_splits)
    dtm=[corpus.doc2bow(i) for i in word_splits]#document term matrix
    pre=list(lda3[dtm])[0]#predicting the new instances
    topics=sorted(pre,key=lambda x:x[1],reverse=True)
    print("Document belong to Topic:",topics[0][0])
if __name__=='__main__':
  Latent_Dirichlet_Allocation=LDA()#creating the object of the class
  data=Latent_Dirichlet_Allocation.load_data()
  dtm,corpus,word_split=Latent_Dirichlet_Allocation.data_preprocessing(data)
  Latent_Dirichlet_Allocation.lda_model1(dtm,corpus,word_split)
  Latent_Dirichlet_Allocation.lda_model2(dtm,corpus,word_split)
  lda3=Latent_Dirichlet_Allocation.lda_model3(dtm,corpus,word_split)
  Latent_Dirichlet_Allocation.predict(lda3)

