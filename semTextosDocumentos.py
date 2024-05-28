import pandas as pd
from dataframeClass import DataFrameReview
from wordCloudGen import WordCloudGenerator

filePath = './chatgpt_reviews.csv'
myDataframe = DataFrameReview(filePath)
periodo = ('2024-05-20','2024-05-21')
dataFrame=myDataframe.getDataFrame(coluna=['content','at','lang'],periodo=periodo,lang=True)
wordCloudGen=WordCloudGenerator(dataFrame)

wordCloudGen.getWordCloud('content','lang',mix=False,langs=['english'])
