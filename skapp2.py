import streamlit as st
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# MongoDB connection setup
uri = "mongodb+srv://koushikvishnud:21mia1054@cluster1054.hh0bz.mongodb.net/?retryWrites=true&w=majority&tls=true&tlsInsecure=true"
client = MongoClient(uri, server_api=ServerApi('1'))

# Access database and collection
db = client['1054_db']  # replace with your database name
collection = db['zomoto_res']  # replace with your collection name

# Function to fetch data from MongoDB with an optional query filter
def fetch_data(query=None):
    """Fetch records from MongoDB based on a specific query."""
    if query:
        # Fetch documents matching the query
        data = list(collection.find(query))
    else:
        # Fetch all documents if no query is provided
        data = list(collection.find())
    return data

# Streamlit page setup
st.set_page_config(page_title="Hotel Recommendation System", layout="centered")

# Background Image URL (update this with your image URL)
background_url = "https://images.squarespace-cdn.com/content/v1/60241cb68df65b530cd84d95/1721062483515-TGYX5YVPF70H4XM8TCSU/Gusteaus15.jpg"

# Custom CSS to set background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .stTextInput input {{
        background-color: rgba(0, 0, 0, 0.8);
        font-size: 18px;
    }}
    .stButton {{
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
    }}
    .stMarkdown {{
        color: #ffffff;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the app
st.title("Hotel Recommendation System")

# User inputs for cuisine, dish, and location preferences
cuisine = st.text_input("Enter your preferred cuisine", "")
dish = st.text_input("Enter a dish you like", "")
location = st.text_input("Enter the location", "")

# Function to recommend hotels based on user input
def recommend_hotels(cuisine, dish, location):
    try:
        if not cuisine or not location:
            st.error("Please enter at least cuisine and location.")
            return

        # Create a search query based on user input
        query = {"$or": [
            {"cuisines": {"$regex": cuisine, "$options": 'i'}},
            {"dish_liked": {"$regex": dish, "$options": 'i'}} if dish else {},
            {"location": {"$regex": location, "$options": 'i'}}
        ]}

        # Fetch data based on the user's query
        data = fetch_data(query)

        # Check if any data was retrieved
        if not data:
            st.warning("No restaurants found matching your criteria.")
            return

        # Preprocessing the dataset (combining relevant columns for better recommendations)
        combined_features = [
            f"{item['cuisines']} {item.get('dish_liked', '')} {item['location']} {item['rest_type']}"
            for item in data
        ]

        # Create a TF-IDF vectorizer model
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(combined_features)

        # Transform the query into the same vector space as the TF-IDF matrix
        query_vec = tfidf.transform([f"{cuisine} {dish} {location}"])

        # Calculate similarity with all restaurants in the dataset
        similarity_scores = cosine_similarity(query_vec, tfidf_matrix)

        # Get the top restaurant recommendations
        top_indices = similarity_scores.argsort()[0][::-1]  # Sort in descending order

        recommendations = []
        seen_hotels = set()  # Set to keep track of unique hotel names

        for idx in top_indices:
            hotel_name = data[idx]['name']
            if hotel_name not in seen_hotels:
                recommendations.append({
                    'name': hotel_name,
                    'rating': data[idx].get('rate', 'N/A'),
                    'url': data[idx].get('url', 'N/A')
                })
                seen_hotels.add(hotel_name)

            # Stop once we have 5 unique recommendations
            if len(recommendations) == 5:
                break

        # Display recommendations in the Streamlit app
        if recommendations:
            st.write("### Top 5 Recommended Hotels:")
            for idx, hotel in enumerate(recommendations, 1):
                st.markdown(
                    f"""
                    <div style="border: 2px solid #ccc; padding: 10px; border-radius: 10px; margin-bottom: 10px; background-color: rgba(0, 0, 0, 0.8);">
                        <p><strong>{idx}. {hotel['name']}</strong></p>
                        <p>Rating: {hotel['rating']}</p>
                        <p><a href="{hotel['url']}" target="_blank">More Info</a></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("No recommendations found based on your preferences.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display button to get recommendations
if st.button("Get Recommendations"):
    recommend_hotels(cuisine, dish, location)

# Add trademark at the bottom of the page
st.markdown(
    """
    <div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); font-size: 14px; color: white;">
        Made by Jeevan, Jagadish, Koushik
    </div>
    """,
    unsafe_allow_html=True
)

