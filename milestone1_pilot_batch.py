import pandas as pd
import numpy as np
import re
import uuid
import logging
from langdetect import detect
from deep_translator import GoogleTranslator
from fuzzywuzzy import fuzz
from collections import Counter
import spacy
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize tools
translator = GoogleTranslator(source='auto', target='en')
nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('all-MiniLM-L6-v2')
llm_classifier = pipeline('text-classification', model='bert-base-multilingual-uncased')

# Predefined categories (extensible)
INITIAL_CATEGORIES = {
    'Size': ['size', 'length', 'width', 'height', 'depth', 'diameter', 'circumference'],
    'Color': ['color', 'colour', 'shade', 'tint', 'hue', 'tone'],
    'Style': ['style', 'design', 'type', 'pattern', 'model', 'version'],
    'Material': ['material', 'fabric', 'texture', 'composition'],
    'Quantity': ['quantity', 'pack', 'count', 'number', 'amount'],
    'Shape': ['shape', 'form', 'cut', 'contour'],
    'Fit': ['fit', 'sleeve', 'waist', 'inseam'],
    'Accessories': ['buckle', 'strap', 'lens', 'chain', 'cord'],
    'Features': ['texture', 'border', 'finish', 'feature'],
    'Components': ['capacity', 'weight', 'component'],
    'Packaging': ['package', 'set', 'box', 'kit'],
    'Customization': ['engraving', 'personalization', 'monogram'],
    'Condition': ['condition', 'status', 'grade'],
    'Location': ['location', 'origin', 'position'],
    'Details': ['model', 'version', 'detail'],
    'Other': ['scent', 'flavor', 'miscellaneous']
}

# Clean text
def clean_text(text):
    if pd.isna(text):
        return ''
    return re.sub(r'[^\w\s]', '', str(text).lower()).strip()

# Translate non-English titles
def translate_to_english(text):
    try:
        lang = detect(text)
        if lang != 'en':
            translated = translator.translate(text)
            return translated if translated else text, lang
        return text, 'en'
    except Exception as e:
        logging.warning(f"Translation failed for '{text}': {e}")
        return text, 'unknown'

# Build dynamic taxonomy with word embeddings
def build_dynamic_taxonomy(titles, model):
    taxonomy = INITIAL_CATEGORIES.copy()
    cleaned_titles = [clean_text(title) for title in titles if pd.notna(title)]
    title_counts = Counter(cleaned_titles)
    common_terms = [term for term, count in title_counts.most_common(200) if len(term) > 2]
    
    # Generate embeddings for common terms
    term_embeddings = model.encode(common_terms)
    
    # Generate embeddings for category keywords
    category_keywords = {cat: model.encode(keywords) for cat, keywords in taxonomy.items()}
    
    for term, embedding in zip(common_terms, term_embeddings):
        best_cat = 'Other'
        best_sim = -1
        for cat, kw_embeddings in category_keywords.items():
            sims = cosine_similarity([embedding], kw_embeddings)[0]
            max_sim = max(sims)
            if max_sim > best_sim and max_sim > 0.65:  # Adjusted threshold for broader inclusion
                best_sim = max_sim
                best_cat = cat
        taxonomy[best_cat].append(term)
    
    inverse_taxonomy = {v.lower(): k for k, vs in taxonomy.items() for v in vs}
    return taxonomy, inverse_taxonomy

# Normalize variant title with enhanced logic
def normalize_variant_title(title, inverse_taxonomy, taxonomy, model):
    if pd.isna(title) or not str(title).strip():
        return 'Unknown', 'Unknown', None, str(uuid.uuid4()), 'Missing or empty title'
    
    original_title = title
    translated_title, lang = translate_to_english(title)
    title_clean = clean_text(translated_title)
    
    primary_attr = 'Unknown'
    secondary_attr = None
    group_id = str(uuid.uuid4())
    notes = f"Language: {lang}"
    
    # Direct match
    if title_clean in inverse_taxonomy:
        primary_attr = inverse_taxonomy[title_clean]
        return original_title, primary_attr, secondary_attr, group_id, notes
    
    # Enhanced fuzzy matching
    best_score = 0
    best_match = None
    for taxonomy_title in inverse_taxonomy.keys():
        score = fuzz.token_sort_ratio(title_clean, taxonomy_title)
        if score > best_score and score > 85:  # Lowered threshold for better matching
            best_score = score
            best_match = taxonomy_title
    
    if best_match:
        primary_attr = inverse_taxonomy[best_match]
        notes += f"; Fuzzy match: {best_match} (score: {best_score})"
        return original_title, primary_attr, secondary_attr, group_id, notes
    
    # NLP-based extraction with spaCy
    doc = nlp(title_clean)
    for token in doc:
        lemma = token.lemma_
        for cat, terms in taxonomy.items():
            if lemma in terms:
                primary_attr = cat
                secondary_attr = token.text
                notes += f"; NLP matched to {cat} with {token.text}"
                return original_title, primary_attr, secondary_attr, group_id, notes
    
    # LLM disambiguation for unresolved cases
    if primary_attr == 'Unknown':
        llm_result = llm_classifier(title_clean[:512])[0]
        if llm_result['score'] > 0.85:  # Adjusted threshold for confidence
            label = llm_result['label'].lower()
            for cat, terms in taxonomy.items():
                if any(label in term for term in terms):
                    primary_attr = cat
                    notes += f"; LLM resolved to {cat}"
                    break
        if primary_attr == 'Unknown':
            notes += "; Unmatched, will be clustered"
    
    return original_title, primary_attr, secondary_attr, group_id, notes

# Generate embeddings
def generate_embeddings(titles, model, batch_size=500):
    embeddings = []
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Cluster titles with hybrid approach
def cluster_titles(titles, embeddings, min_cluster_size=3):
    # Try HDBSCAN first
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='leaf')
    labels = clusterer.fit_predict(embeddings)
    
    # Fallback to KMeans if too many noise points
    if (labels == -1).sum() > len(labels) * 0.4:  # Reduced noise tolerance
        num_clusters = max(1, len(titles) // 8)  # More granular clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
    
    return labels

# Dynamically name clusters
def name_clusters(titles, labels):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(titles)
    feature_names = vectorizer.get_feature_names_out()
    
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    cluster_names = {}
    for label in unique_labels:
        cluster_indices = [i for i, l in enumerate(labels) if l == label]
        cluster_tfidf = tfidf_matrix[cluster_indices].sum(axis=0).A1
        top_words = [feature_names[i] for i in cluster_tfidf.argsort()[-5:][::-1] if cluster_tfidf[i] > 0]
        cluster_names[label] = " ".join(top_words) if top_words else "Generic Cluster"
    
    return cluster_names

# Assign unknowns to closest cluster or create new logical groups
def assign_unknowns(titles, embeddings, labels, cluster_names, model):
    noise_indices = [i for i, l in enumerate(labels) if l == -1]
    if not noise_indices:
        return labels, cluster_names
    
    # Get cluster centroids
    unique_labels = set(labels) - {-1}
    centroids = []
    for label in unique_labels:
        cluster_embeddings = embeddings[labels == label]
        centroid = cluster_embeddings.mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids) if centroids else np.array([])
    
    for idx in noise_indices:
        title_embedding = embeddings[idx]
        if centroids.size > 0:
            similarities = cosine_similarity([title_embedding], centroids)[0]
            max_sim = max(similarities)
            if max_sim > 0.6:  # Adjusted threshold for assignment
                closest_label = list(unique_labels)[np.argmax(similarities)]
                labels[idx] = closest_label
            else:
                # Create new cluster with logical name
                new_label = max(unique_labels, default=-1) + 1
                labels[idx] = new_label
                doc = nlp(titles[idx])
                key_terms = [token.text for token in doc if token.is_alpha and not token.is_stop][:3]
                cluster_names[new_label] = " ".join(key_terms) if key_terms else titles[idx][:20]
                unique_labels.add(new_label)
                centroids = np.vstack([centroids, title_embedding]) if centroids.size > 0 else np.array([title_embedding])
        else:
            # First unknown becomes its own cluster
            new_label = 0
            labels[idx] = new_label
            doc = nlp(titles[idx])
            key_terms = [token.text for token in doc if token.is_alpha and not token.is_stop][:3]
            cluster_names[new_label] = " ".join(key_terms) if key_terms else titles[idx][:20]
            unique_labels.add(new_label)
            centroids = np.array([title_embedding])
    
    return labels, cluster_names

# Process a single title
def process_title(row, inverse_taxonomy, taxonomy, model):
    title = row['variant_title']
    original, primary, secondary, temp_group_id, notes = normalize_variant_title(title, inverse_taxonomy, taxonomy, model)
    return {
        'Original_Title': original,
        'Translated_Title': translate_to_english(title)[0],
        'Normalized_Title': clean_text(title),
        'Primary_Attribute': primary,
        'Secondary_Attribute': secondary,
        'Group_ID': temp_group_id,
        'Notes': notes
    }

def process_pilot_batch(variant_data_file, output_file, sample_size=4000):
    logging.info("Processing Pilot Batch...")
    
    # Load data
    df = pd.read_excel(variant_data_file)
    pilot_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Build dynamic taxonomy
    taxonomy, inverse_taxonomy = build_dynamic_taxonomy(pilot_df['variant_title'], model)
    
    # Process titles in parallel
    with Pool() as pool:
        results = pool.starmap(process_title, [(row, inverse_taxonomy, taxonomy, model) for _, row in pilot_df.iterrows()])
    
    # Create output DataFrame
    result_df = pd.DataFrame(results)
    
    # Cluster all titles, including unknowns
    unique_titles = result_df['Original_Title'].dropna().unique()
    embeddings = generate_embeddings(unique_titles, model)
    labels = cluster_titles(unique_titles, embeddings)
    
    # Name clusters
    cluster_names = name_clusters(unique_titles, labels)
    
    # Assign unknowns to clusters or create new groups
    labels, cluster_names = assign_unknowns(unique_titles, embeddings, labels, cluster_names, model)
    
    # Map labels to group IDs
    title_to_group = {title: f"{cluster_names[label]}_{uuid.uuid4().hex[:8]}" if label != -1 else f"Generic_{uuid.uuid4().hex[:8]}" for title, label in zip(unique_titles, labels)}
    result_df['Group_ID'] = result_df['Original_Title'].map(title_to_group)
    
    # Save output
    result_df.to_excel(output_file, index=False)
    logging.info(f"Pilot batch processed. Output saved to {output_file}")
    return result_df

if __name__ == "__main__":
    process_pilot_batch("variant_data.xlsx", "output/milestone1_pilot_output.xlsx")