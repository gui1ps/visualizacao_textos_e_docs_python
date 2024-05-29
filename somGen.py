import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from minisom import MiniSom
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dataframeClass import DataFrameReview
from matplotlib import pyplot as plt

class SimilaritySOM:

    def __init__(self, dataSet, contentColumnName, langColumnName, pkColumn):
        self.comments = []
        self.setDocuments(dataSet, contentColumnName, langColumnName, pkColumn)

    def setDocuments(self, dataSet, contentColumnName, langColumnName, pkColumn):

        def is_dfPandas(dataFrame):
            if isinstance(dataFrame, pd.DataFrame):
                return True
            else:
                raise ValueError('Um dataframe pandas deve ser fornecido.')

        def iterar():
            if is_dfPandas(dataSet):
                caracteres_especiais = list("!@#$%^&*()_-+={}[]:;'<>?,./|\\~`")
                for index, linha in dataSet.iterrows():
                    try:
                        comentario = linha[contentColumnName]
                        lang = linha[langColumnName]
                        stopwords_lang = nltk.corpus.stopwords.words(lang)
                        comentarioTokens = nltk.word_tokenize(comentario)
                        filtredContentTokens = [token for token in comentarioTokens if token not in stopwords_lang and token not in string.ascii_letters and token not in string.digits and token not in caracteres_especiais]
                        currentText = " ".join(filtredContentTokens)
                        self.comments.append(currentText)
                    except:
                        continue
        iterar()

    def getComments(self):
        return self.comments

    def generate_similarity_som(self, grid_size=(10, 10), sigma=1.0, learning_rate=0.5, num_iterations=1000):
        comments = self.getComments()

        vectorizer = TfidfVectorizer(
            max_df=1.0,
            min_df=1,
            max_features=500
        )
        tfidf_matrix = vectorizer.fit_transform(comments).toarray() 
        som = MiniSom(grid_size[0], grid_size[1], tfidf_matrix.shape[1], sigma=sigma, learning_rate=learning_rate)
        som.train(tfidf_matrix, num_iterations)

        plt.figure(figsize=(8, 6))
        plt.title('Mapa de Calor das Distâncias')
        plt.pcolor(som.distance_map().T, cmap='bone_r')
        plt.colorbar()
        plt.show()

if __name__ == "__main__":
    csv_file = './chatgpt_reviews.csv'
    myDataframe = DataFrameReview(csv_file)
    periodo = ('2024-05-20','2024-05-26')
    dataFrame = myDataframe.getDataFrame(coluna=['userName', 'content', 'lang'], periodo=periodo, lang=True)

    som_gen = SimilaritySOM(dataFrame, 'content', 'lang', 'userName')
    som_gen.generate_similarity_som()