from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from py2neo import Graph
import requests
from config import *
from langchain_community.embeddings import BaichuanTextEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.chat_models import ChatBaichuan
from langchain_anthropic import ChatAnthropic

def get_embeddings_model(): #向量化模型
    model_map = {
        'openai': OpenAIEmbeddings(
            model = os.getenv('OPENAI_EMBEDDINGS_MODEL')
        ),
        # 'baichuan': BaichuanTextEmbeddings()
    }
    return model_map.get(os.getenv('EMBEDDINGS_MODEL'))
# print(get_embeddings_model())
# # exit()


def get_llm_model(): #对话大模型
    model_map = {
        'openai': ChatOpenAI(
            model = os.getenv('OPENAI_LLM_MODEL'),
            temperature = os.getenv('TEMPERATURE'),
            max_tokens = os.getenv('MAX_TOKENS'),
        ) ,
        # #百川大模型
        # 'baichuan': ChatBaichuan(
        #     model = os.getenv('BAICHUAN_LLM_MODEL'),
        #     temperature = os.getenv('TEMPERATURE'),
        # )
        # #Claude，可与媲美甚至超越GPT-4的大语言模型
        # 'claude': ChatAnthropic(
        #     model=os.getenv('CLAUDE_LLM_MODEL'),
        #     temperature = os.getenv('TEMPERATURE'),
        #     max_tokens = os.getenv('MAX_TOKENS'),
        # )
    }
    return model_map.get(os.getenv('LLM_MODEL'))


#json格式
def structured_output_parser(response_schemas):
    text = '''
    请从以下文本中，抽取出实体信息，并按json格式输出，json包含首尾的 "```json" 和 "```"。
    以下是字段含义和类型，要求输出json中，必须包含下列所有字段：\n
    '''
    for schema in response_schemas:
        text += schema.name + ' 字段，表示：' + schema.description + '，类型为：' + schema.type + '\n'
    return text


#文本替换函数
def replace_token_in_string(string, slots):
    for key, value in slots:
        string = string.replace('%'+key+'%', value)#把'%'+key+'%'占位符格式替换成值value
    return string

#连接Neo4j
def get_neo4j_conn():
    return Graph(
        os.getenv('NEO4J_URI'), 
        auth = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
    )

#测试一下
if __name__ == '__main__':
    llm_model = get_llm_model()
    print(llm_model.invoke('编程是什么？'))