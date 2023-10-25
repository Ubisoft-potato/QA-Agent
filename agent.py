from typing import List

import dotenv
import pandas as pd
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import LLMChain
from langchain.chains import LLMMathChain, RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_experimental.tools.python.tool import PythonAstREPLTool


class QaAgent:
    txt_description_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Write a description of given docs and a brief summary of them. \n"
                       "Use the format:\n\n Useful for when you need to answer questions about"
                       " [topic you need to summarize].\n\n"
                       " [brief summary of docs]\n\n"),
            ("human", "{docs}")
        ]
    )

    csv_description_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Write a description of given CSV file. "
                       "Use the format:\n\n Useful for when you need to answer questions about"
                       " [topic you need to summarize]\n stored in pandas dataframe 'df'. \n\n"
                       "Run python pandas operations on 'df' to help you get the right answer.\n\n"
                       "'df' has the following columns: \n\n [df.columns.to_list()]\n\n"
                       "This is the result of `print(df.head())`:\n\n [str(df.head().to_markdown())]\n\n"
                       "The input for this tool should be like:\n\n df['column_name'].mean()\n\n"),
            ("human", "{csv}")
        ]
    )

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        self.csv_description_chain = self.csv_description_prompt | self.llm | StrOutputParser()
        # retriever
        loaders = [
            TextLoader('data/Syllabus.txt'),
            TextLoader('data/Weekly Schedule.txt'),
        ]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
        docs = text_splitter.split_documents(docs)
        subjects_retriever = Chroma.from_documents(docs, OpenAIEmbeddings()).as_retriever(search_kwargs={"k": 2})

        subjects_syllabus_and_weekly_schedule = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=subjects_retriever,
        )

        # create calculator tool
        calculator = LLMMathChain.from_llm(llm=self.llm, verbose=True)

        tools = [
            Tool(
                name="Subjects Syllabus And Weekly Schedule",
                func=subjects_syllabus_and_weekly_schedule.run,
                description=self.gen_docs_description(docs),
            ),
            self.csv_read_tool("data/Grades.csv", "Student Grades Data"),
            self.csv_read_tool("data/Professors.csv", "Professors Data"),
            Tool(
                name="Calculator",
                func=calculator.run,
                description="""
                Useful when you need to do math operations or arithmetic.
                """
            )
        ]

        agent_kwargs = {'prefix': f'You are the headmaster of a school, you need to answer the question about'
                                  f' subjects and professors. You have access to the following tools:'}
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            verbose=True,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            agent_kwargs=agent_kwargs,
        )

    def gen_docs_description(self, docs: List[Document]) -> str:
        stuff_chain = StuffDocumentsChain(
            llm_chain=LLMChain(llm=self.llm, prompt=self.txt_description_prompt),
            document_variable_name="docs"
        )
        return stuff_chain.run(docs)

    def csv_read_tool(self, csv_file, name: str) -> Tool:
        return Tool(
            name=name,
            func=PythonAstREPLTool(locals={"df": pd.read_csv(csv_file)}).run,
            description=self.csv_description_chain.invoke({"csv": open(csv_file, "r").read()}),
        )

    def run(self, query: str) -> str:
        return self.agent.run(query)


if __name__ == '__main__':
    dotenv.load_dotenv()
    agent = QaAgent()

    # print(agent.agent.agent.llm_chain.prompt.template)

    print(agent.run("Which of the Mathematics professors are male?"))
    print("--------------------------------------------------------")
    print(agent.run("What is the syllabus for the subject taught in 3rd period on Monday?"))
    print("--------------------------------------------------------")
    print(agent.run("What is the average age of professors who teach Mathematics in the weekly schedule?"))
    print("--------------------------------------------------------")
    print(agent.run("What is the average grade for the subject taught in 1st period on Tuesday?"))
    print("--------------------------------------------------------")
    print(agent.run("What days and periods is the subject with the lowest average grade taught?"))
    print("--------------------------------------------------------")
    print(agent.run("What is the subject with the lowest average grade? Find its syllabus."))
