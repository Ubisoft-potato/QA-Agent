import dotenv
import pandas as pd
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import LLMMathChain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_experimental.tools.python.tool import PythonAstREPLTool


def csv_read_tool(df: pd.DataFrame, name: str) -> Tool:
    return Tool(
        name=name,
        func=PythonAstREPLTool(locals={"df": df}).run,
        description=f"""
                Useful for when you need to answer questions about {name} stored in pandas
                 dataframe 'df'. 
                Run python pandas operations on 'df' to help you get the right answer.
                'df' has the following columns: 
                {df.columns.to_list()}
                
                This is the result of `print(df.head())`:
                {str(df.head().to_markdown())}
                
                The input for this tool should be like:
                df['column_name'].mean()
                """
    )


class QaAgent:

    def __init__(self):
        llm = ChatOpenAI(temperature=0, model="gpt-4")

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
            llm=llm,
            chain_type="stuff",
            retriever=subjects_retriever,
        )

        # create calculator tool
        calculator = LLMMathChain.from_llm(llm=llm, verbose=True)

        tools = [
            Tool(
                name="Subjects Syllabus And Weekly Schedule",
                func=subjects_syllabus_and_weekly_schedule.run,
                description="""
                Useful for when you need to answer questions about subjects syllabus and subjects weekly schedule with
                professor name.

                <user>: Tell me those professors who teach Mathematics in the weekly schedule!
                <assistant>: I need to check the Subjects Syllabus And Weekly Schedule to answer this question.
                <assistant>: Action: Subjects Syllabus And Weekly Schedule
                <assistant>: Action Input: professors who teach Mathematics in the weekly schedule
                ...
                """
            ),
            csv_read_tool(pd.read_csv("data/Grades.csv"), "Student Grades Data"),
            csv_read_tool(pd.read_csv("data/Professors.csv"), "Professors Data"),
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
            llm=llm,
            verbose=True,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            agent_kwargs=agent_kwargs,
        )

    def run(self, query: str) -> str:
        return self.agent.run(query)


if __name__ == '__main__':
    dotenv.load_dotenv()
    agent = QaAgent()
    # print(agent.run("Which of the Mathematics professors are male?"))
    # print(agent.run("What is the syllabus for the subject taught in 3rd period on Monday?"))
    print(agent.run("What is the average age of professors who teach Mathematics in the weekly schedule?"))
    # print(agent.run("What is the average grade for the subject taught in 1st period on Tuesday?"))
    # print(agent.run("What days and periods is the subject with the lowest average grade taught?"))
    # print(agent.run("What is the subject with the lowest average grade? Find its syllabus."))
