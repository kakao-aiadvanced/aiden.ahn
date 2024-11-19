import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
import json


class RAGPipeline:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def fetch_urls(self, urls: List[str]) -> List[Document]:
        """웹 URL에서 문서를 가져옵니다."""
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

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """문서를 작은 청크로 분할합니다."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(docs)

    def vectorize_documents(self, splits: List[Document]) -> Chroma:
        """문서를 벡터화하고 Chroma DB에 저장합니다."""
        return Chroma.from_documents(documents=splits, embedding=self.embeddings)

    def check_relevancy(self, query: str, retrieved_chunk: str) -> Dict:
        """쿼리와 검색된 청크의 관련성을 확인합니다."""
        relevancy_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            사용자의 query와 제공된 retrieved_chunk가 유사한지 relevancy를 판단하여 
            유사하면 "yes", 아니면 "no"를 반환합니다.
            결과는 아래의 JSON 포맷으로 응답합니다.
            {{
                "relevancy": "yes" or "no"
            }}
            """),
            ("human", "query: {query}\nretrieved_chunk: {retrieved_chunk}")
        ])

        chain = relevancy_prompt | self.llm | JsonOutputParser()
        return chain.invoke({
            "query": query,
            "retrieved_chunk": retrieved_chunk
        })

    def check_hallucination(self, answer: str, context: str) -> Dict:
        """생성된 답변의 환각 여부를 확인합니다."""
        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            제공된 context를 기반으로 생성된 answer의 환각 여부를 검사합니다.
            모든 내용이 context에서 직접적으로 지원되는지 확인하고,
            결과를 다음 JSON 형식으로 반환합니다:
            {{
                "is_hallucination": "yes" or "no",
                "explanation": "판단 근거에 대한 설명"
            }}
            """),
            ("human", "answer: {answer}\ncontext: {context}")
        ])

        chain = hallucination_prompt | self.llm | JsonOutputParser()
        return chain.invoke({
            "answer": answer,
            "context": context
        })

    def generate_answer(self, query: str, context: str) -> str:
        """컨텍스트를 바탕으로 쿼리에 대한 답변을 생성합니다."""
        answer_prompt = hub.pull("rlm/rag-prompt")
        chain = answer_prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "context": context,
            "question": query
        })

    def process_query(self, query: str, urls: List[str]) -> Dict:
        """전체 RAG 파이프라인을 실행합니다."""
        # 1. 문서 준비
        docs = self.fetch_urls(urls)
        splits = self.split_documents(docs)
        vectorstore = self.vectorize_documents(splits)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        # 2. 관련 문서 검색
        retrieved_docs = retriever.invoke(query)
        retrieved_content = "\n".join([doc.page_content for doc in retrieved_docs])

        # 3. 관련성 확인
        relevancy_result = self.check_relevancy(query, retrieved_content)

        if relevancy_result["relevancy"] == "no":
            return {
                "status": "no_relevant_content",
                "message": "관련된 내용을 찾을 수 없습니다."
            }

        # 4. 답변 생성
        answer = self.generate_answer(query, retrieved_content)

        # 5. 환각 검사
        hallucination_check = self.check_hallucination(answer, retrieved_content)

        return {
            "status": "success",
            "answer": answer,
            "relevancy_check": relevancy_result,
            "hallucination_check": hallucination_check,
            "context": retrieved_content
        }


if __name__ == "__main__":
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    rag = RAGPipeline()
    query = "tesla model 3 에 대해 설명해주세요."
    result = rag.process_query(query, urls)
    print(json.dumps(result, indent=2, ensure_ascii=False))
