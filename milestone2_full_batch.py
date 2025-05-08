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
from multiprocessing import Pool

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize tools
translator = GoogleTranslator(source='auto', target='en')
nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('all-MiniLM-L6-v2')

# Attribute hierarchy
ATTRIBUTE_HIERARCHY = {
    'Material': ['Fill Material', 'Cover Material', 'Fabric', 'Leather', 'Metal'],
    'Size': ['Length', 'Width', 'Height', 'Diameter'],
    'Style': ['Design', 'Pattern', 'Shape', 'Finish']
}

# Predefined 16 categories
INITIAL_CATEGORIES = {
    'Size': ['size', 'length', 'width', 'height', 'grandeur', 'talla', 'taglia', 'größe'],
    'Color': ['color', 'colour', 'couleur', 'shade', 'tint', 'hue'],
    'Style': ['style', 'design', 'type', 'finish', 'pattern'],
    'Material': ['material', 'fabric', 'leather', 'metal', 'wood'],
    'Quantity': ['quantity', 'pack', 'number', 'count', 'amount'],
    'Shape': ['shape', 'form', 'cut'],
    'Fit': ['fit', 'sleeve', 'inseam', 'waist'],
    'Product Accessories': ['buckle', 'chain', 'strap', 'lens', 'hardware'],
    'Design Features': ['texture', 'sheen', 'framing', 'border'],
    'Functional Components': ['function', 'backset', 'capacity', 'weight'],
    'Packaging Options': ['package', 'bundle', 'set', 'box'],
    'Customization Choices': ['engraving', 'monogram', 'personalization'],
    'Condition': ['condition', 'status'],
    'Location': ['ships from', 'ship to', 'location'],
    'Specific Product Details': ['model', 'version', 'edition'],
    'Other Attributes': ['scent', 'flavor', 'fragrance']
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
            return translated, lang
        return text, 'en'
    except Exception as e:
        logging.warning(f"Translation failed for '{text}': {e}")
        return text, 'unknown'

# Build dynamic taxonomy
def build_dynamic_taxonomy(titles):
    taxonomy = INITIAL_CATEGORIES.copy()
    cleaned_titles = [clean_text(title) for title in titles if pd.notna(title)]
    title_counts = Counter(cleaned_titles)
    common_terms = [term for term, count in title_counts.most_common(50) if len(term) > 2]
    
    for term in common_terms:
        matched = any(fuzz.ratio(term, v) > 85 for attrs in taxonomy.values() for v in attrs)
        if not matched:
            doc = nlp(term)
            if any(token.lemma_ in ['size', 'length', 'width', 'height'] for token in doc):
                taxonomy['Size'].append(term)
            elif any(token.lemma_ in ['color', 'shade', 'tint'] for token in doc):
                taxonomy['Color'].append(term)
            elif any(token.lemma_ in ['style', 'type', 'design'] for token in doc):
                taxonomy['Style'].append(term)
            else:
                taxonomy['Other Attributes'].append(term)
    
    inverse_taxonomy = {v.lower(): k for k, vs in taxonomy.items() for v in vs}
    return taxonomy, inverse_taxonomy

# Normalize variant title with hierarchy
def normalize_variant_title(title, inverse_taxonomy, taxonomy):
    if pd.isna(title):
        return 'Unknown', 'Unknown', None, None, str(uuid.uuid4()), 'Missing title'
    
    original_title = title
    translated_title, lang = translate_to_english(title)
    title_clean = clean_text(translated_title)
    
    primary_attr = 'Unknown'
    secondary_attr = None
    hierarchical_attr = None
    group_id = str(uuid.uuid4())
    notes = f"Language: {lang}"
    
    # Direct match
    if title_clean in inverse_taxonomy:
        primary_attr = inverse_taxonomy[title_clean]
        for parent, children in ATTRIBUTE_HIERARCHY.items():
            if primary_attr in children:
                hierarchical_attr = parent
                break
        return original_title, primary_attr, secondary_attr, hierarchical_attr, group_id, notes
    
    # Fuzzy matching
    best_score = 0
    best_match = None
    for taxonomy_title in inverse_taxonomy.keys():
        score = fuzz.token_sort_ratio(title_clean, taxonomy_title)
        if score > best_score and score > 85:
            best_score = score
            best_match = taxonomy_title
    
    if best_match:
        primary_attr = inverse_taxonomy[best_match]
        for parent, children in ATTRIBUTE_HIERARCHY.items():
            if primary_attr in children:
                hierarchical_attr = parent
                break
        notes += f"; Fuzzy match: {best_match} (score: {best_score})"
        return original_title, primary_attr, secondary_attr, hierarchical_attr, group_id, notes
    
    # NLP and regex patterns
    patterns = {
        'Size': r'\bsize\b|\blength\b|\bwidth\b|\bheight\b|\bgrandeur\b|\btalla\b',
        'Color': r'\bcolor\b|\bcolour\b|\bcouleur\b|\bshade\b|\btint\b',
        'Style': r'\bstyle\b|\btype\b|\bdesign\b|\bfinish\b',
        'Material': r'\bmaterial\b|\bfabric\b|\bleather\b|\bmetal\b',
        'Quantity': r'\bquantity\b|\bpack\b|\bnumber\b|\bcount\b'
    }
    
    doc = nlp(title_clean)
    words = title_clean.split()
    for attr, pattern in patterns.items():
        if re.search(pattern, title_clean) or any(token.lemma_ in pattern.split('|') for token in doc):
            primary_attr = attr
            if len(words) > 1:
                secondary_attr = ' '.join(words[:-1])
            for parent, children in ATTRIBUTE_HIERARCHY.items():
                if primary_attr in children:
                    hierarchical_attr = parent
                    break
            break
    else:
        notes += "; Unmatched, assigned to generic group"
    
    return original_title, primary_attr, secondary_attr, hierarchical_attr, group_id, notes

# Generate embeddings
def generate_embeddings(titles, batch_size=500):
    embeddings = []
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Cluster titles with generic group naming
def cluster_titles(titles, embeddings):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
    labels = clusterer.fit_predict(embeddings)
    title_to_group = {}
    group_names = [
        "Product Accessories", "Design Features", "Functional Components",
        "Packaging Options", "Customization Choices", "Specific Product Details"
    ]
    
    for idx, (title, label) in enumerate(zip(titles, labels)):
        if label != -1:
            group_id = f"{group_names[label % len(group_names)]}_{uuid.uuid4().hex[:8]}"
        else:
            group_id = f"GenericGroup_{uuid.uuid4().hex[:8]}"
        title_to_group[title] = group_id
    return title_to_group

# Process a single title
def process_title(row, inverse_taxonomy, taxonomy, existing_groups):
    title = row['variant_title']
    original, primary, secondary, hierarchical, temp_group_id, notes = normalize_variant_title(title, inverse_taxonomy, taxonomy)
    
    group_id = temp_group_id
    title_clean = clean_text(title)
    for gid, titles in existing_groups.items():
        for t in titles:
            if fuzz.token_sort_ratio(title_clean, clean_text(t)) > 90:
                group_id = gid
                break
        if group_id != temp_group_id:
            break
    
    return {
        'Original_Title': original,
        'Translated_Title': translate_to_english(title)[0],
        'Normalized_Title': clean_text(title),
        'Primary_Attribute': primary,
        'Secondary_Attribute': secondary,
        'Hierarchical_Attribute': hierarchical,
        'Group_ID': group_id,
        'Notes': notes
    }

def process_full_batch(variant_data_file, pilot_output, output_file):
    logging.info("Milestone 2: Processing Full Batch...")
    
    # Load data
    df = pd.read_excel(variant_data_file)
    pilot_df = pd.read_excel(pilot_output)
    
    # Build dynamic taxonomy
    taxonomy, inverse_taxonomy = build_dynamic_taxonomy(df['variant_title'])
    
    # Get existing groups from pilot
    existing_groups = pilot_df.groupby('Group_ID')['Original_Title'].apply(list).to_dict()
    
    # Process titles in parallel
    with Pool() as pool:
        results = pool.starmap(process_title, [(row, inverse_taxonomy, taxonomy, existing_groups) for _, row in df.iterrows()])
    
    # Create output DataFrame
    result_df = pd.DataFrame(results)
    
    # Cluster unmatched titles
    unmatched = result_df[result_df['Group_ID'].str.startswith('GenericGroup_')]
    if not unmatched.empty:
        unique_titles = unmatched['Original_Title'].dropna().unique()
        embeddings = generate_embeddings(unique_titles)
        title_to_group = cluster_titles(unique_titles, embeddings)
        result_df.loc[result_df['Original_Title'].isin(title_to_group), 'Group_ID'] = result_df['Original_Title'].map(title_to_group)
    
    # Save output
    result_df.to_excel(output_file, index=False)
    logging.info(f"Full batch processed. Output saved to {output_file}")
    return result_df

if __name__ == "__main__":
    process_full_batch("variant_data.xlsx", "output/milestone1_pilot_output.xlsx", "output/milestone2_full_output.xlsx")