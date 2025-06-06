from sentence_transformers import SentenceTransformer
import streamlit as st
@st.cache_resource(show_spinner=False)
def get_weaviate_client():
    import weaviate
    from weaviate.classes.init import Auth
    return weaviate.connect_to_wcs(
    cluster_url="https://vjdlfn7rbum0pszjhpt9g.c0.asia-southeast1.gcp.weaviate.cloud",
    auth_credentials=Auth.api_key("MZHQuXA8XdafltJfaTI0plJV1AZFz1JBqrTN")
)
client = get_weaviate_client()
embedder = SentenceTransformer("all-miniLM-L6-v2")


import pandas as pd
st.set_page_config(page_title="My App", layout="wide")
from PIL import Image
import os
import numpy as np

df = pd.read_csv("C:\\Users\\Bhumikka Pancharane\\OneDrive\\Desktop\\Sem IV Notes and QB\\Datasets\\Fashion_Dataset.csv")

df.head(5)

df.dropna(subset=['p_id','name','img','description','rating','avg_rating'], inplace =True)
df['brand'] = df['brand'].fillna('Unknown')
df['p_attributes'] = df['p_attributes'].fillna('')
df['colour'] = df['colour'].fillna('Unknown')

#Encoding categorical Feature
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
encoded_brand = encoder.fit_transform(df[['brand']]).toarray()
encoded_colour = encoder.fit_transform(df[['colour']]).toarray()

#Extracting text Features
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english', max_features = 100)
desc_features = tfidf.fit_transform(df['description']).toarray()

attr_tfidf = TfidfVectorizer(stop_words='english', max_features = 50)
attr_features = tfidf.fit_transform(df['p_attributes']).toarray()

image_features = np.random.rand(len(df),2048)

#Normalize Features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
desc_features_scaled = scaler.fit_transform(desc_features)
attr_features_scaled = scaler.fit_transform(attr_features)
image_features_scaled = scaler.fit_transform(image_features)


np.save('desc_features.npy', desc_features_scaled)
np.save('attr_features.npy', attr_features_scaled)
np.save('image_features.npy', image_features_scaled)

#combining all the features
combined_features = np.concatenate(
    [desc_features_scaled, attr_features_scaled, image_features_scaled,
    encoded_brand, encoded_colour],axis=1)


#Computer Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
content_similarity = cosine_similarity(combined_features)

# Create synthetic UserID (just index as user id)
new_df = df[['p_id','name','price','colour','brand','img','rating','ratingCount','avg_rating','description','p_attributes']].copy()

new_df['user_id'] = new_df.index
print(new_df.columns)
# Create user-item matrix
user_item_matrix = new_df.pivot_table(index='user_id', columns='p_id', values='rating').fillna(0)

from sklearn.metrics.pairwise import cosine_similarity

item_ratings = df[['avg_rating']].to_numpy()
collab_similarity = cosine_similarity(item_ratings)

#hybrid matrix
def hybrid_score_matrix(w1,w2):
    return w1 * content_similarity + w2 * collab_similarity

def get_hybrid_recommendations(product_index,w1=0.5,w2=0.5,top_n=5):
    if product_index >= len(df):
        product_index = random.randint(0, len(df) - 1)
        
    score_matrix = hybrid_score_matrix(w1,w2)
    sim_scores = list(enumerate(score_matrix[product_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1],reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    recommended_indices = [i[0] for i in sim_scores]
    return df.iloc[recommended_indices][['p_id','name','img','price','rating','avg_rating']]

def find_similar_product_from_prompt(prompt_text):
    try:
        embedded_prompt = embedder.encode(prompt_text).tolist()
        collection = client.collections.get("Product")

        result = collection.query.near_vector(
            near_vector=embedded_prompt,
            limit=1
        )

        if result.objects:
            matched_id = result.objects[0].properties['p_id']
            matches = df[df['p_id'] == matched_id]
            if not matches.empty:
                return matches.index[0]

        return random.randint(0, len(df) - 1)

    except Exception as e:
        st.error(f"Weaviate error: {e}")
        return random.randint(0, len(df) - 1)

def fitness(individual):
    w1,w2 = individual
    if w1 + w2 == 0:
        return 0
    w1 /= (w1+w2)
    w2 /= (w1+w2)
    score_matrix = hybrid_score_matrix(w1,w2)

    avg_top_sim = 0
    num_products = len(df)

    for idx in range(num_products):
        sims = sorted(score_matrix[idx],reverse=True)[1:6]
        avg_top_sim += sum(sims)/len(sims)

    return avg_top_sim / num_products


import random

def init_population(size):
    return [[random.random(),random.random()] for _ in range(size)]

def selection(population, scores, k=3):
    selected = random.choices(list(zip(population, scores)),k=k)
    selected.sort(key=lambda x: x[1], reverse = True)
    return selected[0][0]

def crossover(p1,p2):
    alpha = random.random()
    return [alpha * p1[0] + (1 - alpha)* p2[0],
            alpha * p1[1] + (1 - alpha)* p2[1]]

def mutate(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        idx = random.choice([0,1])
        individual[idx] += random.uniform(-0.1,0.1)
        individual[idx] = max(0,min(1, individual[idx]))
    return individual


def run_genetic_algorithm(generations=1, pop_size=5):
    population = init_population(pop_size)
    best_individual = None
    best_score = -1

    for gen in range(generations):
        scores = [fitness(ind) for ind in population]
        new_population = []

        for _ in range(pop_size):
            p1 = selection(population,scores)
            p2 = selection(population, scores)
            child = crossover(p1,p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population
        max_score = max(scores)
        if max_score > best_score:
            best_score = max_score
            best_individual = population[scores.index(max_score)]

        print(f"Gen {gen+1}: Best Scoore = {best_score:.4f}, Weights = {best_individual}")

    return best_individual

@st.cache_resource
def get_best_weights():
    best_weights = run_genetic_algorithm(generations = 1, pop_size=3)
    best_w1,best_w2 = best_weights
    total = best_w1+best_w2
    return best_w1 /total,best_w2 / total

best_w1, best_w2 = get_best_weights()
# Sidebar

#st.title(":[red]Fashion Recommender :dress:")


st.title(":collision: :blue[Fashion Genie] :collision:")
st.sidebar.markdown(":brain: :red[Fashion Recommender]")
st.sidebar.markdown("Hybrid Recommendation System using Content + Collaborative Filtering optimized by Genetic Algorithm.")

# Toggle between models
search_mode = st.sidebar.radio("Search Mode", ["Hybrid","Content-Based", "Collaborative"])
top_n = st.sidebar.slider("Number of Recommendations", 1, 10, 5)
# Main Page

st.header(":dress: Fashion Recommender 2.0")
st.markdown("Welcome! Select your favorite product below, and we'll show you similar items based on **" + search_mode + "** filtering.")


#Initialize session state
for key in ['fashion_type','occasion','recommendations','selected_product']:
    if key not in st.session_state:
        st.session_state[key] = None if key == 'selected_product' else ""


#Prompts
st.write("Select a  prompt or enter you own below, then click **Search**.")
example_prompts = [
    "casual outfit for a weekend outing",
    "formal dress for a business meeting",
    "trendy streetwear for a party",
    "comfortable loungewear for home",
    "sporty activewear for gym",
    "ethnic wear for weedings"
    ]                  

cols = st.columns(3)
for i, prompt in enumerate(example_prompts):
    if cols[i % 3].button(prompt):
        if " for " in prompt:
            fashion_type, occasion = prompt.split(" for ")
        else:
            fashion_type, occasion = prompt, ""
        st.session_state['fashion_type'] = prompt.split(' for ' )[0]
        st.session_state['occasion'] = prompt.split(' for ')[1] if ' for ' in prompt else ""

#text input fallback
fashion_type = st.text_input(
    "what fashion style or item are you looking for?",
    value = st.session_state.get('fashion_type', ''),
    placeholder = "E.g., Casual outfit, Formal dress, Streetwear" 
)

occasion = st.text_input(
    "What is the occasion?",
    value = st.session_state.get('occasion', ''),
    placeholder = "E.g., Weekend outing, Business meeting, Party"
)
prompt = f"{fashion_type} for {occasion}".strip()
selected_index = find_similar_product_from_prompt(prompt)

#Sidebar for random product
st.sidebar.markdown(":game_die: :rainbow[Get a Random Product]")

if st.sidebar.button("Click for Inspo"):
    idx = np.random.randint(0, len(df))
    st.session_state['selected_product'] = df.iloc[idx]
    st.sidebar.image(df.iloc[idx]['img'], caption = df.iloc[idx]['avg_rating']
                     ,use_container_width = True)


elif st.session_state['selected_product'] is not None:
    st.sidebar.image(st.session_state['selected_product']['img'],
                     caption = st.session_state['selected_product']['avg_rating'],
                     use_container_width = True)


#Unified Search Button

if st.button("Search"):
    st.session_state['fashion_type'] = fashion_type
    st.session_state['occasion'] = occasion

    selected_product = st.session_state.get("selected_product")
    try:
        if selected_product is not None:
            selected_index = df[df['p_id'] == selected_product['p_id']].index[0]
        elif fashion_type or occasion:
            if selected_index is not None and selected_index < len(df):
                st.session_state['selected_product'] = df.iloc[selected_index]
            else:
                st.warning("No product found for your prompt.")
                selected_index = None
        else:
            st.warning("Please select a product or write in the prompt.")
            selected_index = None

        if selected_index is not None:
            recs = get_hybrid_recommendations(selected_index, w1=best_w1, w2=best_w2, top_n=top_n)
            st.session_state['recommendations'] = recs
    except Exception as e:
        st.error(f"Recommendation error: {e}")

#displaying recommendation

st.markdown("---")
st.subheader(":thought_balloon: :orange[Recommended Products]")

recs = st.session_state.get('recommendations')

if isinstance(recs, pd.DataFrame) and not recs.empty:
    st.markdown("### Recommended Products")
    cols = st.columns(len(recs))
    for i, rec in enumerate(recs.itertuples()):
        with cols[i]:
            st.image(rec.img, caption=rec.name, use_container_width=True)
            st.caption(f"₹{rec.price} | ⭐ {rec.avg_rating}")
else:
    st.warning("No recommendations available. Please click 'Search' after selecting a product or entering a prompt.")


        
    

































