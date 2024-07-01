import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from langchain_community.document_loaders.chromium import AsyncChromiumLoader
from langchain_core.embeddings import UniversalSentenceEncoderEmbedding
from langchain_core.vector_store.pinecone_store import PineconeStore

# Load environment variables
load_dotenv()

async def extract_course_details(url, user_agent):
    # Send a GET request to the URL with user agent
    headers = {
        'User-Agent': user_agent
    }
    response = requests.get(url, headers=headers)

    # Check if request was successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract course details
        courses = []
        course_cards = soup.find_all('div', class_='course-card')  # Assuming course cards have this class

        for card in course_cards:
            course_name = card.find('h2', class_='course-name').text.strip()
            course_price = card.find('span', class_='course-price').text.strip()
            course_details = card.find('p', class_='course-details').text.strip()

            # Store the course details in a dictionary
            course = {
                'name': course_name,
                'price': course_price,
                'details': course_details
            }
            courses.append(course)

        return courses
    else:
        print(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
        return None

async def create_embeddings(courses):
    # Initialize Universal Sentence Encoder embedding
    embedding = UniversalSentenceEncoderEmbedding()

    # Embed each course details
    embeddings = []
    for course in courses:
        embedding_vector = await embedding.embed_text(course['details'])
        embeddings.append(embedding_vector)

    return embeddings

def main():
    # URL of the course page
    url = "https://brainlox.com/courses/category/technical"

    # Read user agent from environment variables
    user_agent = os.getenv('USER_AGENT')

    # Step 1: Extract course details
    courses = asyncio.run(extract_course_details(url, user_agent))

    if not courses:
        print("No courses extracted. Exiting.")
        return

    # Step 2: Create embeddings
    embeddings = asyncio.run(create_embeddings(courses))

    # Print or store embeddings (e.g., in a vector store like Pinecone)
    print("Embeddings created:", embeddings)

    # Example: Store embeddings in Pinecone (replace with your Pinecone API and index name)
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_index_name = "course_embeddings"

    pinecone_store = PineconeStore(api_key=pinecone_api_key)
    for idx, embedding in enumerate(embeddings):
        pinecone_store.insert_item(pinecone_index_name, idx, embedding)

    print("Embeddings stored in Pinecone.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
