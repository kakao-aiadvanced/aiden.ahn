from langchain import hub
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict
from vector_store import retriever
from web_search import search
import streamlit as st


class RAGPipeline:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name)

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

    # def check_usefulness(self, question: str, generation: str) -> Dict:
    #     # Prompt
    #     system = """You are a grader assessing whether an
    #         answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
    #         useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""
    #
    #     prompt = ChatPromptTemplate.from_messages(
    #         [
    #             ("system", system),
    #             ("human", "question: {question}\n\n answer: {generation} "),
    #         ]
    #     )
    #
    #     answer_grader = prompt | self.llm | JsonOutputParser()
    #     return answer_grader.invoke({"question": question, "generation": generation})

    def generate_answer(self, query: str, context: str) -> str:
        """컨텍스트를 바탕으로 쿼리에 대한 답변을 생성합니다."""
        answer_prompt = hub.pull("rlm/rag-prompt")
        chain = answer_prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "context": context,
            "question": query
        })

    def process_query(self, query: str) -> Dict:
        """전체 RAG 파이프라인을 실행합니다."""

        # 1. 관련 문서 검색
        retrieved_content = retriever.retrieve(query)

        # 2. 관련성 확인
        relevancy_result = self.check_relevancy(query, retrieved_content)

        if relevancy_result["relevancy"] == "no":
            # 관련성이 없는 경우
            retrieved_content = search(query)
            relevancy_result = self.check_relevancy(query, retrieved_content)
            if relevancy_result["relevancy"] == "no":
                return {
                    "status": "failed",
                    "message": "No relevant content found"
                }

        # 4. 답변 생성
        answer = self.generate_answer(query, retrieved_content)

        # 5. 환각 검사
        hallucination_check = self.check_hallucination(answer, retrieved_content)
        if hallucination_check["is_hallucination"] == "yes":
            answer = self.generate_answer(query, retrieved_content)
            hallucination_check = self.check_hallucination(answer, retrieved_content)
            if hallucination_check["is_hallucination"] == "yes":
                return {
                    "status": "failed",
                    "message": "Answer is hallucinated"
                }

        return {
            "status": "success",
            "answer": answer,
            "relevancy_check": relevancy_result,
            "hallucination_check": hallucination_check,
            "context": retrieved_content
        }


if __name__ == "__main__":
    rag = RAGPipeline()

    # 제목
    st.title("이것은 마지막 실습")

    # 입력 필드
    st.text_input("자 이제 여기에 뭔가를 입력해 봅시다:", key="query")

    # 버튼
    if st.button("Submit"):
        q = str(st.session_state.query)
        result = rag.process_query(q)

        st.subheader("status")
        st.write(result["status"])
        if result["status"] == "failed":
            st.write(result["message"])
        else:
            st.write(result["relevancy_check"])
            st.write(result["hallucination_check"])
            st.subheader("Answer")
            st.write(result["answer"])
            st.subheader("Context")
            st.write(result["context"])

