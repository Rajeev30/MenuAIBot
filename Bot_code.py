import os
import spacy
import pandas as pd
import numpy as np
import requests
import wikipedia
import faiss

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from langchain_groq import ChatGroq
from google import genai
from google.genai import types
from pydantic import Field 

from langchain.llms.base import LLM
from langchain.schema import Document, BaseRetriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

os.environ["GENIE_API_KEY"] = "APIkey"
os.environ["OPENAI_API_KEY"] = "APIkey"
client = genai.Client(api_key=os.environ["GENIE_API_KEY"])

YELP_API_KEY = "APIkey"
BASE_URL = "https://api.yelp.com/v3"

HEADERS = {
    "Authorization": f"Bearer {YELP_API_KEY}",
    "Content-Type": "application/json"
}
os.system("pip install --upgrade pip")
os.system("python -m spacy download en_core_web_sm")

# import en_core_web_sm
# nlp = en_core_web_sm.load()

#nlp = spacy.load("en_core_web_sm")

groq_api = "API"
llm_groq = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=groq_api)

csv_file_path = "/Sample_Ingredients_File.csv"
df_excel = pd.read_excel("/Sample_Ingredients_File.xlsx")
df_excel.to_csv(csv_file_path, index=False)

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

ingredient_texts = []
for _, row in df_excel.iterrows():
    ingredient_entry = (
        f"Ingredient: {row['ingredient_name']}. "
        f"Description: {row['menu_description'] if pd.notna(row['menu_description']) else ''}. "
        f"Address: {row['address1']}, {row['city']}, {row['state']} {row['zip_code']}"
    )
    ingredient_texts.append(str(ingredient_entry))  

embeddings = model.encode(ingredient_texts)
embeddings_np = np.array(embeddings).astype(np.float32)
dimension = embeddings_np.shape[1] 
index = faiss.IndexFlatL2(dimension)

normalized_embeddings = normalize(embeddings_np, axis=1, norm="l2")
index.add(normalized_embeddings)


def search_businesses(keyword, location, limit=5):
    """
    Searches for businesses based on a keyword and location.
    Returns chatbot-friendly responses.
    """
    url = f"{BASE_URL}/businesses/search"
    params = {
        "term": keyword,
        "location": location,
        "limit": limit
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        return f"Error: {response.json().get('error', {}).get('description', 'Unknown error')}"

    businesses = response.json().get("businesses", [])
    if not businesses:
        return "No businesses found for your search."

    results = []
    for business in businesses:
        name = business.get("name", "Unknown")
        business_id = business.get("id", "N/A")
        rating = business.get("rating", "N/A")
        price = business.get("price", "N/A")
        address = ", ".join(business.get("location", {}).get("display_address", []))
        url = business.get("url", "")

        business_info = (
            f"*{name}*\n"
            f"Rating: {rating}/5\n"
            f"Price: {price}\n"
            f"Address: {address}\n"
            f"[Yelp Page]({url})\n\n"
            f"*Recent Reviews:*\n{get_reviews(business_id)}"
        )
        results.append(business_info)
    return "\n\n".join(results)

def get_reviews(business_id):
    """
    Fetches up to 3 reviews for a business using the business ID or alias.
    """
    url = f"{BASE_URL}/businesses/{business_id}/reviews"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return f"Error fetching reviews: {response.json().get('error', {}).get('description', 'Unknown error')}"
    reviews = response.json().get("reviews", [])
    if not reviews:
        return "No reviews available for this business."

    formatted_reviews = []
    for review in reviews:
        user = review.get("user", {}).get("name", "Anonymous")
        text = review.get("text", "No review text available.")
        rating = review.get("rating", "N/A")
        review_text = f"📝 *{user}* ({rating}/5): {text}"
        formatted_reviews.append(review_text)
    return "\n".join(formatted_reviews)

def compare_average_price(category1, category2, location):
    """
    Compares the average price level between two restaurant categories (using Yelp).
    """
    def get_price_levels(cat):
        url = f"{BASE_URL}/businesses/search"
        params = {"term": cat, "location": location, "limit": 10}
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code != 200:
            return []
        return [b.get("price", "N/A") for b in response.json().get("businesses", []) if b.get("price")]

    prices_cat1 = get_price_levels(category1)
    prices_cat2 = get_price_levels(category2)

    if not prices_cat1 or not prices_cat2:
        return "Not enough data available for comparison."

    avg_price1 = sum(len(p) for p in prices_cat1) / len(prices_cat1)
    avg_price2 = sum(len(p) for p in prices_cat2) / len(prices_cat2)

    return (
        f"*Price Comparison*\n\n"
        f"*{category1.capitalize()} restaurants*: {'💲' * int(avg_price1)}\n"
        f"*{category2.capitalize()} restaurants*: {'💲' * int(avg_price2)}"
    )

def track_menu_trends(keyword, location):
    """
    Fetches recent mentions of an ingredient or dish from Yelp reviews.
    """
    url = f"{BASE_URL}/businesses/search"
    params = {"term": keyword, "location": location, "limit": 5}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        return f"Error: {response.json().get('error', {}).get('description', 'Unknown error')}"

    businesses = response.json().get("businesses", [])
    if not businesses:
        return "No businesses found mentioning this ingredient or dish."

    results = []
    for business in businesses:
        business_id = business.get("id")
        reviews = get_reviews(business_id)
        info = (
            f"*{business.get('name')}*\n"
            f"Address: {', '.join(business.get('location', {}).get('display_address', []))}\n"
            f"[Yelp Page]({business.get('url')})\n\n"
            f"*Recent Mentions:*\n{reviews}"
        )
        results.append(info)

    return "\n\n".join(results)

def extract_nouns(query):
    doc = nlp(query)
    relevant_noun_phrases = [np.text for np in doc.noun_chunks]
    return relevant_noun_phrases 

def retrieve_relevant_chunks(query, index, k=5):
    query_embedding = model.encode([query]).astype(np.float32)
    query_embedding_normalized = normalize(query_embedding, axis=1, norm='l2')
    distances, indices = index.search(query_embedding_normalized, k)
    
    retrieved_chunks = []
    for i in range(k):
        chunk_text = ingredient_texts[indices[0][i]]
        restaurant_name = df_excel.iloc[indices[0][i]]['restaurant_name']
        retrieved_chunks.append(f"Restaurant: {restaurant_name} | Info: {chunk_text}")
    return retrieved_chunks, distances

def retrieve_relevant_chunks_with_wikipedia(query, restaurant_index, wikipedia_index, k, wikipedia_texts):
    restaurant_retrieved_chunks, restaurant_distances = retrieve_relevant_chunks(query, restaurant_index, k)
    
    query_embedding = model.encode([query]).astype(np.float32)
    query_embedding_normalized = normalize(query_embedding, axis=1, norm='l2')
    wikipedia_distances, wikipedia_indices = wikipedia_index.search(query_embedding_normalized, k)

    wikipedia_retrieved_chunks = [wikipedia_texts[i] for i in wikipedia_indices[0]]
    
    retrieved_chunks = restaurant_retrieved_chunks + wikipedia_retrieved_chunks
    combined_distances = restaurant_distances[0].tolist() + wikipedia_distances[0].tolist()
    return retrieved_chunks, combined_distances

def embed_wikipedia_data(noun_data, model):
    """
    Takes a dict {term: summary}, encodes it, and builds a separate FAISS index on the fly.
    """
    wikipedia_texts = list(noun_data.values())
    wikipedia_embeddings = model.encode(wikipedia_texts)
    wikipedia_embeddings_np = np.array(wikipedia_embeddings).astype(np.float32)
    wiki_index = faiss.IndexFlatL2(wikipedia_embeddings_np.shape[1])
    wiki_index.add(normalize(wikipedia_embeddings_np, axis=1, norm='l2'))

    return wiki_index, wikipedia_embeddings_np, wikipedia_texts

def fetch_wikipedia_summary(term):
    try:
        results = wikipedia.search(term, results=1)
        if not results:
            return f"No Wikipedia results found for '{term}'."
        summary = wikipedia.summary(results[0])
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"DisambiguationError: Multiple results for '{term}'. Options: {e.options}"
    except Exception as e:
        return f"Error retrieving Wikipedia: {str(e)}"


def get_most_relevant_term(query, noun_phrases):
    """
    Calls Gemini to parse the query into 6 fields:
    Food, Location, Wikipedia, Cat1, Cat2, timeword
    """
    prompt = (
        f"This is the user query: '{query}'\n\n"
        "Extract and return the following:\n"
        "- **Food-related term** (dish, ingredient, cuisine)\n"
        "- **Location** (if not available, return California by default)\n"
        "- **Wikipedia searchable phrase** (for historical/cultural reference)\n"
        "- **Categories** (if different cuisine types are found, Example: vegan and mexican)\n"
        "- **Time-series word** (Example: Trend)\n\n"
        "**Format the response as:**\n"
        "'Food: <food_item>, Location: <location>, Wikipedia: <wiki_term>, Cat1: <category1>, Cat2: <category2>, timeword: <trend>'\n\n"
        "Return only the structured response without any additional text."
    )
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=50
            )
        )
        response_text = response.text.strip()

        food_item = None
        location = None
        wiki_term = None
        cat1 = None
        cat2 = None
        timeword = None

        for part in response_text.split(", "):
            if "Food:" in part:
                food_item = part.split(":")[1].strip()
            elif "Location:" in part:
                location = part.split(":")[1].strip()
            elif "Wikipedia:" in part:
                wiki_term = part.split(":")[1].strip()
            elif "Cat1:" in part or "category1:" in part:
                cat1 = part.split(":")[1].strip()
            elif "Cat2:" in part or "category2:" in part:
                cat2 = part.split(":")[1].strip()
            elif "timeword:" in part:
                timeword = part.split(":")[1].strip()

        return food_item, location, wiki_term, cat1, cat2, timeword

    except Exception as e:
        print("Error during Gemini API call for relevant term parsing:", e)
        return None, None, None, None, None, None


# def create_references(wikipedia_data, faiss_retrieved_chunks, faiss_distances, langchain_data):
#     """
#     You can expand this to build a references string if needed.
#     """
#     return "References not fully implemented. (Customize in create_references.)"


def rag(query, k=5):
    """
    Your original RAG pipeline, with Yelp, Wikipedia, FAISS retrieval, etc.
    """
    # nouns = extract_nouns(query)
    nouns = []

    food, location, wikiterm, cat1, cat2, timeword = get_most_relevant_term(query, nouns)
    yelpdata = []

    if food or location:
        restaurant_results = search_businesses(food, location)
        yelpdata.append(restaurant_results)
    if cat1 and cat2 and location:
        categorical_result = compare_average_price(cat1, cat2, location)
        yelpdata.append(categorical_result)
    if timeword and food and location:
        timeseriesdata = track_menu_trends(food, location)
        yelpdata.append(timeseriesdata)

    noun_data = fetch_wikipedia_summary(wikiterm) if wikiterm else "No Wikipedia term found."
    wiki_index, wiki_emb_np, wiki_texts = embed_wikipedia_data({wikiterm: noun_data}, model)

    retrieved_chunks, faiss_distances = retrieve_relevant_chunks_with_wikipedia(
        query, index, wiki_index, k, wiki_texts
    )

    # references = create_references(
    #     wikipedia_data={wikiterm: noun_data}, 
    #     faiss_retrieved_chunks=retrieved_chunks,
    #     faiss_distances=faiss_distances,
    #     langchain_data=retrieved_chunks
    # )

    context = f"Context: {retrieved_chunks}.\n\nWikipedia Data: {noun_data}\n\nYelp Data: {yelpdata}"
    
    return context


class GeminiLLM(LLM):
    """
    A minimal wrapper around Gemini to let LangChain treat it as an LLM
    without Pydantic errors.
    """
    model_name: str = Field(default="gemini-2.0-flash")
    temperature: float = Field(default=0.7)
    
    _client = None 

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        **kwargs
    ):
        """Initialize Gemini LLM with custom fields."""
        super().__init__(model_name=model_name, temperature=temperature, **kwargs)
        self._client = genai.Client(api_key=os.environ["GENIE_API_KEY"])

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "custom-gemini"

    def _call(self, prompt: str, stop=None) -> str:
        """
        The method LangChain calls to get the model's response.
        We use self._client to call Gemini here.
        """
        try:
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=500
                )
            )
            return resp.text
        except Exception as e:
            print("GeminiLLM _call error:", e)
            return "Error from Gemini"
        
        
class RAGRetriever(BaseRetriever):
    """
    A custom LangChain retriever that calls your `rag()` function internally.
    Instead of returning chunked Documents, we return a single Document with all the context.
    """
    def _get_relevant_documents(self, query: str):
        final_answer = rag(query) 
        doc = Document(page_content=final_answer)
        return [doc]

    async def _aget_relevant_documents(self, query: str):
        return self.get_relevant_documents(query)


def build_conversational_chain():
    """
    Creates a ConversationalRetrievalChain with a custom system prompt.
    """
    rag_retriever = RAGRetriever()

    gemini_chain_llm = GeminiLLM(model_name="gemini-2.0-flash", temperature=0.7)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    system_prompt = PromptTemplate(
        input_variables=["context", "question"],  
        template="""
        You are a Food Guide. The context below has information from a Faiss database, Wikipedia, and Yelp api.\n
        Context: {context}

        \nThis is the query: {question}

        Read the query and answer it based on the data you have above. Check your answer twice. You have the data above, don't forget. If there is no relevant data available, politely say that the data isn't sufficient.
            """
        )

    chain = ConversationalRetrievalChain.from_llm(
        llm=gemini_chain_llm,
        retriever=rag_retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
        combine_docs_chain_kwargs={"prompt": system_prompt, "document_variable_name": "context"} 
    )

    return chain

