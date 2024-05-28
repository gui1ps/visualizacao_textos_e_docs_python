import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import PyPDF2

class SimilarityScatterPlotGen:
    def __init__(self, documents):
        self.setDocuments(documents)
        
    def setDocuments(self, documents):
        def is_pdf(docname):
            if isinstance(docname, str):
                return docname.lower().endswith('.pdf')
            else:
                raise ValueError('Document name must be a string')
        
        def read_pdf(document):
            text = []
            with open(document,'rb') as pdf:
                leitor=PyPDF2.PdfReader(pdf)
                numPaginas=len(leitor.pages)
                for i in range(numPaginas):
                    paginaAtual=leitor.pages[i]
                    text.append(paginaAtual.extract_text() or '')  # Adicionado 'or ""' para evitar None
                pdf.close()
            return '\n'.join(text)
        
        if isinstance(documents, tuple) and len(documents) > 1:
            self.documents = []
            self.doc_names = []
            for doc in documents:
                if is_pdf(doc):
                    doc_name = doc.split('/')[-1].split('.')[0]
                    self.doc_names.append(doc_name)
                    doc_content = read_pdf(doc)
                    self.documents.append(doc_content)
                else:
                    raise ValueError(f"{doc} não é um arquivo PDF.")
        else:
            raise ValueError("Uma tupla com pelos menos dois documentos deve ser fornecida.")
    
    def getDocNames(self):
        return self.doc_names
    
    def getDocs(self):
        return self.documents
    
    def generate_similarity_scatter_plot(self):
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_df=1.0,               
            min_df=1,                 
            max_features=500         
        )
        documents = self.getDocs()
        document_names = self.getDocNames()
        tfidf_matrix = vectorizer.fit_transform(documents)
        cosine_sim_matrix = cosine_similarity(tfidf_matrix)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(cosine_sim_matrix)

        plt.figure(figsize=(8, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1])

        for i, doc in enumerate(document_names):
            plt.annotate(doc, (pca_result[i, 0], pca_result[i, 1]))

        plt.title("Gráfico de Dispersão de Similaridade de Textos")
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.show()


if __name__ == "__main__":
    path1='./1.pdf'
    path2='./2.pdf'
    path3='./3.pdf'
    documents = (path1, path2, path3)
    scatter_plot_gen = SimilarityScatterPlotGen(documents)
    scatter_plot_gen.generate_similarity_scatter_plot()
