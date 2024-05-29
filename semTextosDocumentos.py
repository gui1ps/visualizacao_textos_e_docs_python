import pandas as pd
from dataframeClass import DataFrameReview
from wordCloudGen import WordCloudGenerator
from SimilarityScatterPlotGenerator import SimilarityScatterPlotGen

filePath = './chatgpt_reviews.csv'
myDataframe = DataFrameReview(filePath)
periodo = ('2024-05-20','2024-06-20')
dataFrame=myDataframe.getDataFrame(coluna=['userName','content','at','lang'],periodo=periodo,lang=True)
wordCloudGen=WordCloudGenerator(dataFrame)
similarityScatterPlotGen=SimilarityScatterPlotGen(dataFrame,'content','lang','userName')

similarityScatterPlotGen.generate_similarity_scatter_plot()
wordCloudGen.getWordCloud('content','lang',mix=False,langs=['english'])