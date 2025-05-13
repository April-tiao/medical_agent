from utils import *
from config import *
from prompt import *

import os
from langchain.chains import LLMChain, LLMRequestsChain #LLMRequestsChain谷歌搜索用到的
from langchain.prompts import PromptTemplate #加载模板
from langchain.vectorstores.chroma import Chroma #加载向量
from langchain.vectorstores.faiss import FAISS #文档向量化，也可以替换别的包
from langchain.schema import Document #创建文档
from langchain.agents import ZeroShotAgent, AgentExecutor, Tool ,create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser #格式化输出
from langchain import hub

class Agent(): #写agent的具体逻辑
    def __init__(self): #实例化时就把文档数据加载进来
        self.vdb = Chroma(
            persist_directory = os.path.join(os.path.dirname(__file__), './data/db'), 
            embedding_function = get_embeddings_model()
        )

    #定义Tool函数，回答日常交际问题
    def generic_func(self, x, query): #query用户说的话
        # print(query)

        prompt = PromptTemplate.from_template(GENERIC_PROMPT_TPL)
        
        llm_chain = LLMChain(
            llm = get_llm_model(), 
            prompt = prompt,
            verbose = os.getenv('VERBOSE') #查看中间过程
        )
        return llm_chain.invoke(query)['text']
    
    #召回并总结答案
    def retrival_func(self,x, query):
        # 召回并过滤文档
        documents = self.vdb.similarity_search_with_relevance_scores(query, k=5) #找出最相关的5条
        #print(documents)#查看结构
        # print(query_result)#list形式
        query_result = [doc[0].page_content for doc in documents if doc[1]>0.7] #分数大于0.7的保留；
                                                                                #doc[0]是documents内容，doc[1]是分数


        # 填充提示词并总结答案
        prompt = PromptTemplate.from_template(RETRIVAL_PROMPT_TPL)
        retrival_chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result) else '没有查到'
            #\n换行
        }
        return retrival_chain.invoke(inputs)['text']

    def graph_func(self, x,  query):
        # 命名实体识别
        response_schemas = [
            ResponseSchema(type='list', name='disease', description='疾病名称实体'),
            ResponseSchema(type='list', name='symptom', description='疾病症状实体'),
            ResponseSchema(type='list', name='drug', description='药品名称实体'),
        ]

        output_parser = StructuredOutputParser(response_schemas=response_schemas)#实例化 格式化输出的对象
        format_instructions = structured_output_parser(response_schemas)

        ner_prompt = PromptTemplate(
            template = NER_PROMPT_TPL,
            partial_variables = {'format_instructions': format_instructions},
            input_variables = ['query']
        )

        ner_chain = LLMChain(
            llm = get_llm_model(),
            prompt = ner_prompt,
            verbose = os.getenv('VERBOSE')
        )

        result = ner_chain.invoke({
            'query': query
        })['text']
        
        ner_result = output_parser.parse(result)
        #print(ner_result)
        #该函数的测试问题
        # print(agent.graph_func('感冒一般是由什么引起的？'))
        # print(agent.graph_func('感冒吃什么药好得快？可以吃阿莫西林吗？'))
    

        #替换模板占位符
        # 命名实体识别结果，填充模板
        graph_templates = []
        for key, template in GRAPH_TEMPLATE.items():
            slot = template['slots'][0] #拿到词槽如'slots': ['disease']
            slot_values = ner_result[slot] #值在命名实体识别的结果里
            for value in slot_values:
                graph_templates.append({
                    'question': replace_token_in_string(template['question'], [[slot, value]]),#[[slot, value]]拼接成2维列表格式
                    'cypher': replace_token_in_string(template['cypher'], [[slot, value]]),
                    'answer': replace_token_in_string(template['answer'], [[slot, value]]),
                })

            # print(graph_templates)#查看效果
            # exit()
    

        if not graph_templates:
            return 

        #筛选相关问题
        #计算问题相似度，筛选最相关问题
        graph_documents = [
            Document(page_content=template['question'], metadata=template)
            for template in graph_templates
        ]
        db = FAISS.from_documents(graph_documents, get_embeddings_model())
        graph_documents_filter = db.similarity_search_with_relevance_scores(query, k=3)
        # print(graph_documents_filter)#查看筛选后的结果


        # 执行CQL，拿到结果
        query_result = []
        neo4j_conn = get_neo4j_conn()
        for document in graph_documents_filter:
            question = document[0].page_content #{[1]是分数
            cypher = document[0].metadata['cypher']
            answer = document[0].metadata['answer']
            try:
                result = neo4j_conn.run(cypher).data()
                if result and any(value for value in result[0].values()):
                    answer_str = replace_token_in_string(answer, list(result[0].items()))
                    query_result.append(f'问题：{question}\n答案：{answer_str}')
            except:
                pass
        # print(query_result)#作为内容传给大模型

        # 总结答案
        prompt = PromptTemplate.from_template(GRAPH_PROMPT_TPL)
        graph_chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        inputs = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result) else '没有查到'
        }
        return graph_chain.invoke(inputs)['text']

    # 搜索工具函数
    def search_func(self, query):
        prompt = PromptTemplate.from_template(SEARCH_PROMPT_TPL)
        llm_chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        llm_request_chain = LLMRequestsChain(
            llm_chain = llm_chain,
            requests_key = 'query_result'
        )
        inputs = {
            'query': query,
            'url': 'https://www.so.com/s?q='+query.replace(' ', '+')
        }
        return llm_request_chain.invoke(inputs)['output']
    
    #agent加速  
    # def query(self, query):
    #     tool = self.parse_tools(tools, query)
    #     return tool.func(query)

    def query(self, query):

        # print(query)#原始问题
        tools = [
            Tool.from_function(
                name = 'generic_func',
                func = lambda x:self.generic_func(x, query),
                description = '可以解答通用领域的知识，例如打招呼，问你是谁等问题',
            ),
            Tool.from_function(
                name = 'retrival_func',
                func = lambda x:self.retrival_func(x, query),
                description = '用于回答寻医问药网相关问题',
            ),
            Tool(
                name = 'graph_func',
                func = lambda x:self.graph_func(x, query),
                description = '用于回答疾病、症状、药物等医疗相关问题', #模板里用到的实体
            ),
            Tool(
                name = 'search_func',
                func = self.search_func,#大模型总结，不会死循环
                description = '其他工具没有正确答案时，通过搜索引擎，回答通用类问题',
            ),
        ]
        # tool = self.parse_tools(tools, query)
        # return tool.func(query)
   
        # prefix = """请用中文，尽你所能回答以下问题。您可以使用以下工具："""
        # suffix = """Begin!  
        # History: {chat_history}
        # Question: {input}
        # Thought:{agent_scratchpad}"""

        # agent_prompt = ZeroShotAgent.create_prompt(
        #     tools=tools,
        #     prefix=prefix,
        #     suffix=suffix,
        #     input_variables=['input', 'agent_scratchpad', 'chat_history']
        # )
        # llm_chain = LLMChain(llm=get_llm_model(), prompt=agent_prompt)
        # agent = ZeroShotAgent(llm_chain=llm_chain)

        # memory = ConversationBufferMemory(memory_key='chat_history')
        # agent_chain = AgentExecutor.from_agent_and_tools( #agent 定义完后用一个执行器调用
        #     agent = agent, 
        #     tools = tools, 
        #     memory = memory, 
        #     verbose = os.getenv('VERBOSE')
        # )
        # return agent_chain.run({'input': query})

        prompt = hub.pull('hwchase17/react-chat') #别人写好的提示词

        prompt = PromptTemplate.from_template(REACT_CHAT_PROMPT_TPL)
        prompt.template = '请用中文回答问题！Final Answer 必须尊重 Obversion 的结果，不能改变语义。\n\n' + prompt.template
        agent = create_react_agent(llm=get_llm_model(), tools=tools, prompt=prompt)
        memory = ConversationBufferMemory(memory_key='chat_history')
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent = agent, 
            tools = tools, 
            memory = memory, 
            handle_parsing_errors = True,
            verbose = os.getenv('VERBOSE')
        )
        return agent_executor.invoke({"input": query})['output']  

    def parse_tools(self, tools, query):
        prompt = PromptTemplate.from_template(PARSE_TOOLS_PROMPT_TPL)
        llm_chain = LLMChain(
            llm = get_llm_model(),
            prompt = prompt,
            verbose = os.getenv('VERBOSE')
        )
        # 拼接工具描述参数
        tools_description = ''
        for tool in tools:
            tools_description += tool.name + ':' + tool.description + '\n'
        result = llm_chain.invoke({'tools_description':tools_description, 'query':query})
        # 解析工具函数
        for tool in tools:
            if tool.name == result['text']:
                return tool
        return tools[0]


#调用测试
if __name__ == '__main__':
    agent = Agent() #实例化agent
    print(agent.query('你好'))
    # print(agent.query('寻医问药网获得过哪些投资？'))
    # print(agent.query('鼻炎和感冒是并发症吗？'))
    # print(agent.query('鼻炎怎么治疗？'))
    # print(agent.query('烧橙子可以治感冒吗？'))

    # print(agent.generic_func('','你叫什么名字？')) #调用测试？

    # print(agent.retrival_func('介绍一下寻医问药网')) 
    print(agent.retrival_func('寻医问药网的客服电话是多少？'))#问题来自csv文件

    # print(agent.graph_func('','感冒一般是由什么引起的？'))
    # print(agent.graph_func('','感冒吃什么药好得快？可以吃阿莫西林吗？'))
    # print(agent.graph_func('','感冒和鼻炎是并发症吗？'))

    # 调用测试
    # print(agent.search_func('python编程要学什么？'))    




