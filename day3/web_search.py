from tavily import TavilyClient

tavily = TavilyClient(api_key='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')


def search(query):
    response = tavily.search(query=query, max_results=3)
    context = [{"url": obj["url"], "content": obj["content"]} for obj in response['results']]
    return context
