import pandas as pd
import numpy as np
from langdetect import detect

class DataFrameReview:
    def __init__(self, filePath):
        self.setDataFrame(filePath)
    
    def setDataFrame(self, filePath):
        def filtrar(filePath):
            if(isinstance(filePath, str)):
                return filePath.split('.')[-1]
        
        tratamentos = {
            'csv': pd.read_csv,
            'excel': pd.read_excel,
            'json': pd.read_json,
            'html': pd.read_html,
            'parquet': pd.read_parquet
        }
        
        extensao = filtrar(filePath)
        
        if extensao in tratamentos:
            self._dataFrame = tratamentos[extensao](filePath)
            if 'at' in self._dataFrame.columns:
                self._dataFrame['at'] = pd.to_datetime(self._dataFrame['at'])
        else:
            raise ValueError(f"Os arquivos do tipo {extensao} não são suportados")
        
    def getDataFrame(self, coluna=None, periodo=None, lang=False):
        
        if(lang):
            languages = {
                "af": "afrikaans",
                "ar": "arabic",
                "bg": "bulgarian",
                "bn": "bengali",
                "ca": "catalan",
                "cs": "czech",
                "cy": "welsh",
                "da": "danish",
                "de": "german",
                "el": "greek",
                "en": "english",
                "es": "spanish",
                "et": "estonian",
                "fa": "persian",
                "fi": "finnish",
                "fr": "french",
                "gu": "gujarati",
                "he": "hebrew",
                "hi": "hindi",
                "hr": "croatian",
                "hu": "hungarian",
                "id": "indonesian",
                "it": "italian",
                "ja": "japanese",
                "kn": "kannada",
                "ko": "korean",
                "lt": "lithuanian",
                "lv": "latvian",
                "mk": "macedonian",
                "ml": "malayalam",
                "mr": "marathi",
                "ne": "nepali",
                "nl": "dutch",
                "no": "norwegian",
                "pa": "punjabi",
                "pl": "polish",
                "pt": "portuguese",
                "ro": "romanian",
                "ru": "russian",
                "sk": "slovak",
                "sl": "slovenian",
                "so": "somali",
                "sq": "albanian",
                "sv": "swedish",
                "sw": "swahili",
                "ta": "tamil",
                "te": "telugu",
                "th": "thai",
                "tl": "tagalog",
                "tr": "turkish",
                "uk": "ukrainian",
                "ur": "urdu",
                "vi": "vietnamese",
                "zh-cn": "chinese (simplified)",
                "zh-tw": "chinese (traditional)"
            }
            def detectarIdioma(conteudo):
                try:
                    return languages[str(detect(conteudo))]
                except:
                    return 'undefined'
           
        if coluna is None:
            if periodo is None:
                if(lang):
                    self._dataFrame['lang']=self._dataFrame['content'].apply(detectarIdioma)
                return pd.DataFrame(self._dataFrame)
            else:
                periodo = [np.datetime64(periodo[0]), np.datetime64(periodo[1])]
                dataFrameFiltrado = pd.DataFrame(self._dataFrame[(self._dataFrame['at']>periodo[0]) & (self._dataFrame['at']<periodo[1])])
                if(lang):
                    dataFrameFiltrado['lang']=dataFrameFiltrado['content'].apply(detectarIdioma)
                return dataFrameFiltrado
        else:
            if periodo is None:
                if(lang):
                    self._dataFrame['lang']=self._dataFrame['content'].apply(detectarIdioma)
                return pd.DataFrame(self._dataFrame[coluna])
            else:
                periodo = [np.datetime64(periodo[0]), np.datetime64(periodo[1])]
                dataFrameFiltrado = pd.DataFrame(self._dataFrame[(self._dataFrame['at']>periodo[0]) & (self._dataFrame['at']<periodo[1])])
                if(lang):
                    dataFrameFiltrado['lang']=dataFrameFiltrado['content'].apply(detectarIdioma)
                return dataFrameFiltrado[coluna]
    
if __name__ == "__main__":
    csv_file = './chatgpt_reviews.csv'
    df_review = DataFrameReview(csv_file)
    
    periodo = ('2024-05-20','2024-05-21')
    
    df_filtrado = df_review.getDataFrame().head(1)
    print(df_filtrado)
    df_filtrado = df_review.getDataFrame(coluna=['content']).head(1)
    print(df_filtrado)
    df_filtrado = df_review.getDataFrame(coluna=['content','at'],periodo=periodo).head(1)
    print(df_filtrado)
    df_filtrado = df_review.getDataFrame(coluna=['content','at','lang'],periodo=periodo,lang=True).head(1)
    print(df_filtrado)