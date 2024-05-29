import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dataframeClass import DataFrameReview
import mplcursors

class SimilarityScatterPlotGen:

    def __init__(self, dataSet,contentColumnName,langColumnName,pkColumn):
        self.names = [] 
        self.comments = []
        self.setDocuments(dataSet,contentColumnName,langColumnName,pkColumn)
        
    def setDocuments(self, dataSet,contentColumnName,langColumnName,pkColumn):

        def is_dfPandas(dataFrame):
            if isinstance(dataFrame, pd.DataFrame):
                return True
            else:
                raise ValueError('Um dataframe pandas deve ser fornecido.')
            
        def iterar():
            if(is_dfPandas(dataSet)):
                caracteres_especiais=list("!@#$%^&*()_-+={}[]:;'<>?,./|\\~`")
                for index,linha in dataSet.iterrows():
                    try:
                        comentario=linha[contentColumnName]
                        lang=linha[langColumnName]
                        name=linha[pkColumn]
                        stopwords=nltk.corpus.stopwords.words(lang)
                        comentarioTokens=nltk.word_tokenize(comentario)
                        filtredContentTokens=[token for token in comentarioTokens if token not in stopwords and token not in string.ascii_letters and token not in string.digits and token not in caracteres_especiais]
                        currentText=" ".join(filtredContentTokens)
                        self.names.append(name)
                        self.comments.append(currentText)
                    except:
                        continue
        iterar()
        

    def getNames(self):
        return self.names
    
    def getComments(self):
        return self.comments
    
    def generate_similarity_scatter_plot(self):
        vectorizer = TfidfVectorizer(
            max_df=1.0,               
            min_df=1,                 
            max_features=500         
        )
        comments = self.getComments()
        names = self.getNames()
        colors=plt.cm.tab10(np.linspace(0, 1, len(set(names))))
        tfidf_matrix = vectorizer.fit_transform(comments)
        cosine_sim_matrix = cosine_similarity(tfidf_matrix)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(cosine_sim_matrix)

        plt.figure(figsize=(8, 6))
        scatter=plt.scatter(pca_result[:, 0], pca_result[:, 1])

        mplcursors.cursor(scatter, hover=True).connect("add", lambda sel: sel.annotation.set_text(names[sel.target.index]))

        plt.title("Gráfico de Dispersão de Similaridade de Textos")
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.show()


if __name__ == "__main__":
    csv_file = './chatgpt_reviews.csv'
    myDataframe = DataFrameReview(csv_file)
    periodo = ('2024-05-20','2024-05-26')
    dataFrame=myDataframe.getDataFrame(coluna=['userName','content','lang'],periodo=periodo,lang=True)

    scatter_plot_gen = SimilarityScatterPlotGen(dataFrame,'content','lang','userName')
    scatter_plot_gen.generate_similarity_scatter_plot()
