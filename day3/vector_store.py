import bs4
from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStore:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=model_name)
        urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]
        docs = self.fetch_urls(urls)
        splits = self.split_documents(docs)
        vectorstore = self.vectorize_documents(splits)
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

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

    def retrieve(self, query: str) -> str:
        """쿼리에 대해 관련 문서를 검색합니다."""
        retrieved_docs = self.retriever.invoke(query)
        retrieved_content = "\n".join([doc.page_content for doc in retrieved_docs])
        return retrieved_content


retriever = VectorStore()
