import shutup
shutup.please()

from utils import *
import os
from glob import glob #遍历文件夹下的所有文件
from langchain.vectorstores.chroma import Chroma #向量化
from langchain.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader,UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter #文档分割，分片的长度
load_dotenv

def doc2vec():
    # 定义文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300, #一般500，文档小则小一些
        chunk_overlap = 50
    )
    # 读取并分割文件
    current_file_path = os.path.abspath(__file__)
    current_dir_path = os.path.dirname(current_file_path)
    # 拼接目录路径，生成绝对路径
    dir_path = os.path.join(current_dir_path, 'data', 'inputs')+ os.sep #+ os.sep保证结尾有斜杠
    # print(dir_path)
    
    documents = []
    for file_path in glob(dir_path + '*.*'):
        loader = None
        if '.csv' in file_path:
            loader = CSVLoader(file_path, encoding='utf-8')
        # if '.pdf' in file_path:
        #     loader = PyMuPDFLoader(file_path)
        # if '.txt' in file_path:
        #     loader = TextLoader(file_path, encoding='utf-8')
        # if '.docx' in file_path:
        #     loader = UnstructuredWordDocumentLoader(file_path, encoding='utf-8')
        if loader:
            # documents += loader.load_and_split(text_splitter)
            documents.extend(loader.load_and_split(text_splitter))
       
    # print(documents) #查看是否成功
    # exit()
    # print(get_embeddings_model())
   

    # 向量化并存储
    if documents:
        # 获取向量化模型
        embeddings_model = get_embeddings_model()
        vdb = Chroma.from_documents(
            documents = documents, 
            embedding = embeddings_model, #向量化的模型
            persist_directory=os.path.join(current_dir_path, 'data', 'db') #存储到某个目录，在data文件夹中生成db文件
        )
        vdb.persist()

if __name__ == '__main__':
    doc2vec()








