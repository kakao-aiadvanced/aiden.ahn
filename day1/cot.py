from openai import OpenAI
client = OpenAI()


completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": """
Solve the following problem step-by-step: 23 + 47

Step-by-step solution:
1. Add the units place.
2. Add the tens place.
3. Combine the tens place and the units place.

Answer:
            """
        }
    ]
)
print(completion.choices[0].message)
