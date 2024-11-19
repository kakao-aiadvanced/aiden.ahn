import bs4  # 웹 스크래핑을 위한 BeautifulSoup4 라이브러리
from langchain import hub  # LangChain
from langchain_chroma import Chroma  # Chroma 벡터 데이터베이스
from langchain_community.document_loaders import WebBaseLoader  # 웹사이트에서 문서를 로드하기 위한 도구
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser  # 문자열 출력을 파싱하기 위한 도구
from langchain_core.runnables import RunnablePassthrough  # 데이터 파이프라인을 구성하기 위한 도구
from langchain_openai import OpenAIEmbeddings  # OpenAI의 임베딩 모델을 사용하기 위한 모듈
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 긴 텍스트를 작은 조각으로 나누기 위한 도구
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


def fetch_urls(urls) -> list[Document]:
    docs = []
    for url in urls:

        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs.extend(loader.load())
    return docs


def split_documents(docs) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits


def vectorize_documents(splits) -> Chroma:
    return Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))



llm = ChatOpenAI(model="gpt-4o-mini")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
#
# # 나눈 텍스트 조각들을 Chroma 데이터베이스에 저장합니다
# # OpenAI의 text-embedding-3-small 임베딩 모델을 사용하여 텍스트를 벡터로 변환합니다
# vectorstore = vectorize_documents(split_documents(fetch_urls(urls)))
#
# # 저장된 문서에서 정보를 검색하기 위한 retriever를 생성합니다
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
#
# # LangChain 허브에서 RAG(Retrieval Augmented Generation) 프롬프트를 가져옵니다
# prompt = hub.pull("rlm/rag-prompt")



system = """
    사용자의 query 와 제공된 retrieved_chunk 가 유사한지 relevancy 를 판단하여 유사하면 "yes", 아니면 "no" 를 반환합니다.
    결과는 아래의 JSON 포맷으로 응답합니다.
    {{
        "relevancy": "yes" or "no"
    }}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "query: {query}\n retrieved_chunk: {retrieved_chunk}")
    ]
)

chain = prompt | llm | JsonOutputParser()

result = chain.invoke({"query": "agent memory", "retrieved_chunk": split_documents(fetch_urls(urls))})

print(result)


# 이번엔 실패하는 케이스
result = chain.invoke({"query": "agent memory", "retrieved_chunk": [
    Document("The game of association football is played in accordance with the Laws of the Game, a set of rules that has been in effect since 1863 and maintained by the IFAB since 1886. "),
    Document("The game is played with a football that is 68–70 cm (27–28 in) in circumference. "),
    Document("The two teams compete to score goals by getting the ball into the other team's goal (between the posts, under the bar, and fully across the goal line)."),
    Document("When the ball is in play, the players mainly use their feet, but may also use any other part of their body, such as their head, chest and thighs, except for their hands or arms, to control, strike, or pass the ball. Only the goalkeepers may use their hands and arms, and that only within the penalty area."),

]})

print(result)

