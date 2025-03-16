import time
from redisvl.extensions.llmcache import SemanticCache
from vectorizer import GoogleGenAIVectorizer
import os
from google import genai
from google.genai import types

os.environ["GOOGLE_API_KEY"]="<API_KEY>"
llmcache = SemanticCache(
    name="llmcache",                                               # underlying search index name
    redis_url="redis://localhost:6379",                            # redis connection url string
    distance_threshold=0.1,                                        # semantic cache distance threshold
    vectorizer=GoogleGenAIVectorizer(model="text-embedding-004"),  # vectorizer object
    ttl=30,                                                        # time-to-live for cache entries in seconds
)


def answer_question(question: str) -> str:
    results=llmcache.check(prompt=question)
    if results:
        return results[0]["response"]
    else:
        client=genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        response=client.models.generate_content(
            model="gemini-1.5-flash-001",
            contents=[question]
        )
        llmcache.store(prompt=question, response=response.text)
        return response.text

start = time.time()
# asking a question -- LLM response time
question = "What was the name of the first US President?"
answer = answer_question(question)
end = time.time()
print(f"Question: {question}\nAnswer: {answer}\nTime taken: {end-start:.2f} seconds")

start = time.time()
# asking a question -- semantic cache response time
question = "Who was the first US President?" # same question but different wording
answer = answer_question(question)
end = time.time()
print(f"Question: {question}\nAnswer: {answer}\nTime taken: {end-start:.2f} seconds")

time.sleep(30)  # wait for cache to expire
start = time.time()
# asking a question -- LLM response time
question = "What was the name of the first US President?"
answer = answer_question(question)
end = time.time()
print(f"Question: {question}\nAnswer: {answer}\nTime taken: {end-start:.2f} seconds")