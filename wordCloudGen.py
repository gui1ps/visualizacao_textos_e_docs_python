import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from dataframeClass import DataFrameReview

nltk.download('punkt')
nltk.download('stopwords')

class WordCloudGenerator:
    
    def __init__(self,dataFrame):
        self.setDataSet(dataFrame)
    
    def setDataSet(self,dataFrame):
        if(isinstance(dataFrame,pd.DataFrame)):
            self._dataSet=dataFrame
        else:
            raise ValueError('Você está tentando configurar um dataframe que não é instância do pandas. Apenas dataframes do pandas são aceitos.')
    
    def getWordCloud(self,contentColumnName,langColumnName,mix=False,langs=list()):
        
        def plotar(content):
            wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(content)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.show()
            
        def iterar(mainData,contentColumnName,langColumnName):
            if(isinstance(mainData,pd.DataFrame)):
                myBigText=''
                for index,linha in mainData.iterrows():
                    comentario=linha[contentColumnName]
                    lang=linha[langColumnName]
                    try:
                        stopwords=nltk.corpus.stopwords.words(lang)
                        contentTokens=nltk.word_tokenize(comentario)
                        filtredContentTokens=[token for token in contentTokens if token not in stopwords]
                        currentText=" ".join(filtredContentTokens)
                        myBigText+=f' {currentText}'
                    except:
                        continue
                return myBigText
        
        if mix:
            mainData=self._dataSet[[contentColumnName,langColumnName]]
            text=iterar(mainData,contentColumnName,langColumnName)
            plotar(text)
        else:
            mainData=self._dataSet[[contentColumnName,langColumnName]]
            condition=mainData[langColumnName].isin(langs)
            filtredMainData=mainData.loc[condition,[contentColumnName,langColumnName]]
            text=iterar(filtredMainData,contentColumnName,langColumnName)
            plotar(text)
            
            
        
if __name__=='__main__':
    
    csv_file = './chatgpt_reviews.csv'
    myDataframe = DataFrameReview(csv_file)
    periodo = ('2024-05-20','2024-05-26')
    
    dataFrame=myDataframe.getDataFrame(coluna=['content','at','lang'],periodo=periodo,lang=True)

    wordCloudGen=WordCloudGenerator(dataFrame)
    wordCloudGen.getWordCloud('content','lang',mix=False,langs=['english'])
