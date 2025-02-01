import streamlit as st
import numpy as np
import numpy.linalg as la
import pickle
import os
import gdown
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import math

# Initialize session state variables
if "text_search" not in st.session_state:
    st.session_state.text_search = "Roses are red, trucks are blue, and Seattle is grey right now"
if "categories" not in st.session_state:
    st.session_state.categories = "Flowers Colors Cars Weather Food"

# Function to compute cosine similarity
def cosine_similarity(x, y):
    """
    Compute exponentiated cosine similarity between two vectors.
    """
    dot_product = np.dot(x, y)
    magnitude_x = la.norm(x)
    magnitude_y = la.norm(y)
    cosine_sim = dot_product / (magnitude_x * magnitude_y)
    return math.exp(cosine_sim)

# Function to compute averaged GloVe embeddings
def averaged_glove_embeddings_gdrive(sentence, word_index_dict, embeddings, model_type=50):
    """
    Compute averaged GloVe embeddings for a sentence.
    """
    embedding = np.zeros(int(model_type.split("d")[0]))
    words = sentence.split()
    valid_word_count = 0

    for word in words:
        if word.lower() in word_index_dict:
            word_index = word_index_dict[word.lower()]
            embedding += embeddings[word_index]
            valid_word_count += 1

    if valid_word_count > 0:
        embedding /= valid_word_count
    return embedding

# Function to get sorted cosine similarity
def get_sorted_cosinesimilarity(_, embeddings_metadata):
    """
    Get sorted cosine similarity between input sentence and categories.
    """
    categories = st.session_state.categories.split(" ")
    if not categories:
        st.warning("No categories provided. Please enter categories.")
        return []

    cosine_sim = {}

    if embeddings_metadata["embedding_model"] == "glove":
        # GloVe embeddings
        word_index_dict = embeddings_metadata["word_index_dict"]
        embeddings = embeddings_metadata["embeddings"]
        model_type = embeddings_metadata["model_type"]

        # Get input sentence embedding
        input_embedding = averaged_glove_embeddings_gdrive(
            st.session_state.text_search, word_index_dict, embeddings, model_type
        )

        # Compute embeddings for each category
        category_embeddings = {
            category: averaged_glove_embeddings_gdrive(category, word_index_dict, embeddings, model_type)
            for category in categories
        }

        # Compute cosine similarity for each category
        for category, category_embedding in category_embeddings.items():
            cosine_sim[category] = cosine_similarity(input_embedding, category_embedding)

    else:
        # Sentence Transformer embeddings
        model_name = embeddings_metadata.get("model_name", "all-MiniLM-L6-v2")

        # Load or update category embeddings
        if f"cat_embed_{model_name}" not in st.session_state:
            get_category_embeddings(embeddings_metadata)

        category_embeddings = st.session_state[f"cat_embed_{model_name}"]

        # Get input sentence embedding
        input_embedding = get_sentence_transformer_embeddings(
            st.session_state.text_search, model_name=model_name
        )

        # Compute cosine similarity for each category
        for category in categories:
            if category not in category_embeddings:
                # Update category embeddings if not found
                category_embeddings[category] = get_sentence_transformer_embeddings(category, model_name=model_name)
            cosine_sim[category] = cosine_similarity(input_embedding, category_embeddings[category])

    # Sort cosine similarities in descending order
    sorted_cosine_sim = sorted(cosine_sim.items(), key=lambda x: x[1], reverse=True)
    return sorted_cosine_sim

# Main function for the Streamlit app
if __name__ == "__main__":
    st.sidebar.title("GloVe Twitter")
    st.sidebar.markdown(
        """
    GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Pretrained on 
    2 billion tweets with vocabulary size of 1.2 million. Download from [Stanford NLP](http://nlp.stanford.edu/data/glove.twitter.27B.zip). 

    Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. *GloVe: Global Vectors for Word Representation*.
    """
    )

    # Sidebar to choose the GloVe model type
    model_type = st.sidebar.selectbox("Choose the model", ("25d", "50d", "100d"), index=1)

    # Main app title and description
    st.title("Search Based Retrieval Demo")
    st.subheader("Pass in space-separated categories you want this search demo to be about.")
    st.session_state.categories = st.text_input(
        label="Categories", key="categories", value=st.session_state.categories
    )

    st.subheader("Pass in an input word or even a sentence")
    st.session_state.text_search = st.text_input(
        label="Input your sentence",
        key="text_search",
        value=st.session_state.text_search,
    )

    # Download GloVe embeddings if they don't exist
    embeddings_path = f"embeddings_{model_type}_temp.npy"
    word_index_dict_path = f"word_index_dict_{model_type}_temp.pkl"
    if not os.path.isfile(embeddings_path) or not os.path.isfile(word_index_dict_path):
        with st.spinner("Downloading GloVe embeddings..."):
            try:
                download_glove_embeddings_gdrive(model_type)
            except Exception as e:
                st.error(f"Failed to download GloVe embeddings: {e}")
                st.stop()

    # Load GloVe embeddings
    try:
        word_index_dict, embeddings = load_glove_embeddings_gdrive(model_type)
    except Exception as e:
        st.error(f"Failed to load GloVe embeddings: {e}")
        st.stop()

    # Find closest word to an input word
    if st.session_state.text_search:
        # GloVe embeddings
        st.write("### GloVe Embedding Results")
        embeddings_metadata = {
            "embedding_model": "glove",
            "word_index_dict": word_index_dict,
            "embeddings": embeddings,
            "model_type": model_type,
        }
        with st.spinner("Computing cosine similarity for GloVe..."):
            sorted_cosine_sim_glove = get_sorted_cosinesimilarity(
                st.session_state.text_search, embeddings_metadata
            )

        # Sentence Transformer embeddings
        st.write("### Sentence Transformer Embedding Results")
        embeddings_metadata = {"embedding_model": "transformers", "model_name": "all-MiniLM-L6-v2"}
        with st.spinner("Computing cosine similarity for Sentence Transformer..."):
            sorted_cosine_sim_transformer = get_sorted_cosinesimilarity(
                st.session_state.text_search, embeddings_metadata
            )

        # Results and Plot Pie Chart
        st.subheader(f"Closest word between: {st.session_state.categories} as per different embeddings")
        plot_alatirchart(
            {
                f"glove_{model_type}": sorted_cosine_sim_glove,
                "sentence_transformer_384": sorted_cosine_sim_transformer,
            }
        )

        st.write("")
        st.write("Demo developed by [Your Name](https://www.linkedin.com/in/your_id/ - Optional)")
