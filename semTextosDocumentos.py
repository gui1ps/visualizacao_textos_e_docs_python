from dataframeClass import DataFrameReview
from wordCloudGen import WordCloudGenerator
from SimilarityScatterPlotGenerator import SimilarityScatterPlotGen
from BarPlot import BarPlotGen
from somGen import SimilaritySOM

filePath = './chatgpt_reviews.csv'
pdfpath='./Interactive_Data_Visualization_ Foundations_Techniques_and_Applications_Second_Edit.pdf'
myDataframe = DataFrameReview(filePath)

periodo = ('2024-05-20','2024-06-20')
dataFrame=myDataframe.getDataFrame(coluna=['userName','content','at','lang'],periodo=periodo,lang=True)
wordCloudGen=WordCloudGenerator(dataFrame)
similarityScatterPlotGen=SimilarityScatterPlotGen(dataFrame,'content','lang','userName')
bpg=BarPlotGen(pdfpath,'portuguese')
som_gen = SimilaritySOM(dataFrame, 'content', 'lang', 'userName')

wordCloudGen.getWordCloud('content','lang',mix=False,langs=['english'])
bpg.getBarPlot()
similarityScatterPlotGen.generate_similarity_scatter_plot()
som_gen.generate_similarity_som()