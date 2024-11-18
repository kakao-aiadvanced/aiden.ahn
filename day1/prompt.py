from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},

        {
            "role": "user",
            "content": "banana-바나나\napple-사과\norange-오렌지\nschool-학교\nfriend-친구"
        },
        {
            "role": "user",
            "content": "dog"
        },
    ]
)

print(completion.choices[0].message)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": """

Request: "월급이 50000 이상인 근로자" SQL Query: SELECT * FROM employees WHERE salary > 50000;
Request: "재고가 0인 상품" SQL Query: SELECT * FROM products WHERE stock = 0;
Request: "수학 점수가 90점보다 높은 학생" SQL Query: SELECT name FROM students WHERE math_score > 90;

Convert the following natural language requests into SQL queries.
Request: "Find the average salary of employees in the marketing department." SQL Query:
        """}
    ]
)
print(completion.choices[0].message)