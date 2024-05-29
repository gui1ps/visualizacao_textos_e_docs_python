import matplotlib.pyplot as plt
from collections import Counter
from PyPDF2 import PdfReader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')

class BarPlotGen:
    def __init__(self,path,lang):
        self.words=[]
        self.setDoc(path,lang)
        
    def getWords(self):
        return self.words
    
    def getPathEx(self,path):
        if(isinstance(path,str)):
            return path.split('.')[-1]
    
    def setDoc(self,path,lang):
        caracteres_especiais=list("!@#$%^&*()_-+={}[]:;'<>?,./|\\~`")
        if(self.getPathEx(path)=='pdf'):
            reader=PdfReader(path)
            numPages=len(reader.pages)
            for i in range(numPages):
                try:
                    page=reader.pages[i]
                    text=page.extract_text()
                    tokenized_text=nltk.word_tokenize(text)
                    stopwords=nltk.corpus.stopwords.words(lang)
                    filtredContentTokens=[token for token in tokenized_text if token not in stopwords and token not in string.ascii_letters and token not in string.digits and token not in caracteres_especiais]
                    self.words.extend(filtredContentTokens)
                except Exception as error:
                    print(error)    
        else:
            raise ValueError(f'O arquivo {path} não é um pdf')
        
    def getBarPlot(self):
        wordsList=self.getWords()
        word_counts = Counter(wordsList)
        common_words = word_counts.most_common(10)

        words, counts = zip(*common_words)

        plt.figure(figsize=(10, 5))
        plt.bar(words, counts)
        plt.xlabel('Palavras')
        plt.ylabel('Contagem')
        plt.title('Frequência de palavras')
        plt.show()
                
if __name__=="__main__":
    
    path='./Interactive_Data_Visualization_ Foundations_Techniques_and_Applications_Second_Edit.pdf'
    bpg=BarPlotGen(path,'portuguese')
    bpg.getBarPlot()