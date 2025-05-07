import pandas as pd
import re
from fuzzywuzzy import fuzz
import uuid
from langdetect import detect
from deep_translator import GoogleTranslator
import numpy as np
from collections import Counter
import logging
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize translator and NLP
translator = GoogleTranslator(source='auto', target='en')
nlp = spacy.load('en_core_web_sm')

# Function to clean text
def clean_text(text):
    if pd.isna(text):
        return ''
    return re.sub(r'[^\w\s]', '', str(text).lower()).strip()

# Function to detect and translate non-English titles
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

# Build dynamic taxonomy with NLP
def build_dynamic_taxonomy(titles):
    taxonomy = {
        'Size': ['size', 'sizes', 's-m-l', 's-m-l-xl', 'grandeur', 'taille', 'talla', 'taglia', 'größe', 'us size', 'uk size', 'eu size', 'ring size', 'necklace size', 'bracelet size', 'waist size', 'shoe size', 'clothing size', 'bedding size', 'plant size', 'canvas size', 'art size', 'pot size', 'rug size', 'watch size', 'sleeve length', 'inseam', 'height', 'width', 'length', 'depth', 'diameter', 'tamaño', 'strap size', 'neck size'],
        'Color': ['color', 'colour', 'couleur', 'colore', 'color code', 'primary color', 'secondary color', 'frame color', 'lens color', 'metal color', 'gem color', 'stone color', 'fabric color', 'upholstery color', 'hair color', 'emitting color', 'color1', 'color2', 'gold color', 'combined color'],
        'Material': ['material', 'fabric', 'leather', 'metal', 'wood', 'gemstone', 'stone', 'glass', 'composition', 'upholstery', 'material/color'],
        'Style': ['style', 'type', 'design', 'pattern', 'fit', 'finish', 'sheen', 'texture', 'shape', 'diamond shape', 'cut', 'model', 'version', 'edition', 'collection', 'category', 'framing options', 'border options', 'engraving options', 'product type', 'production method', 'door knob shape'],
        'Quantity': ['quantity', 'pack size', 'bundle size', 'select quantity', 'count', 'amount', 'number', 'servings', 'package quantity', 'select a sock size', 'select compression strength'],
        'Scent': ['scent', 'fragrance', 'flavor', 'flavour', 'scent profile', 'perfume'],
        'Hardware': ['hardware', 'buckle', 'buckle color', 'clasp', 'clasp option', 'hook', 'hook size', 'chain', 'chain type', 'chain length', 'rope', 'rope length', 'hanger', 'hanging option'],
        'Letter': ['letter', 'letra', 'initial', 'monogram', 'name', 'font'],
        'Other': ['condition', 'ships from', 'ship from', 'origin', 'season', 'function', 'feature', 'option', 'options', 'add-on', 'add ons', 'personalization', 'personalisation', 'engraving', 'status', 'packaging', 'packaging type', 'warranty', 'purpose', 'preference', 'location']
    }
    
    cleaned_titles = [clean_text(title) for title in titles if pd.notna(title)]
    title_counts = Counter(cleaned_titles)
    common_terms = [term for term, count in title_counts.most_common(50) if len(term) > 2]
    
    for term in common_terms:
        matched = False
        for attr, variants in taxonomy.items():
            if any(fuzz.ratio(term, v) > 85 for v in variants):
                matched = True
                break
        if not matched:
            doc = nlp(term)
            if any(token.lemma_ in ['shape', 'form'] for token in doc):
                taxonomy['Style'].append(term)
            elif 'co2' in term.lower():
                taxonomy['Other'].append(term)
            else:
                taxonomy['Other'].append(term)
    
    inverse_taxonomy = {v.lower(): k for k, vs in taxonomy.items() for v in vs}
    return taxonomy, inverse_taxonomy

# Normalize variant title with NLP
def normalize_variant_title(title, inverse_taxonomy, taxonomy):
    if pd.isna(title):
        return 'Unknown', 'Other', None, str(uuid.uuid4()), 'Missing title'
    
    original_title = title
    translated_title, lang = translate_to_english(title)
    title_clean = clean_text(translated_title)
    
    primary_attr = 'Other'
    secondary_attr = None
    group_id = str(uuid.uuid4())
    notes = f"Language: {lang}"
    
    if title_clean in inverse_taxonomy:
        primary_attr = inverse_taxonomy[title_clean]
        return original_title, primary_attr, secondary_attr, group_id, notes
    
    best_score = 0
    best_match = None
    for taxonomy_title in inverse_taxonomy.keys():
        score = fuzz.token_sort_ratio(title_clean, taxonomy_title)
        if score > best_score and score > 85:
            best_score = score
            best_match = taxonomy_title
    
    if best_match:
        primary_attr = inverse_taxonomy[best_match]
        if 'color' in best_match.lower() and primary_attr != 'Color':
            secondary_attr = 'Color'
            notes += f"; Fuzzy match: {best_match} (score: {best_score})"
        elif 'size' in best_match.lower() and primary_attr != 'Size':
            secondary_attr = 'Size'
            notes += f"; Fuzzy match: {best_match} (score: {best_score})"
        return original_title, primary_attr, secondary_attr, group_id, notes
    
    regex_patterns = {
        'Size': r'\bsize\b|\bheight\b|\bwidth\b|\blength\b|\bdepth\b|\bdiameter\b|\bs-m-l\b|\bxs\b|\bs\b|\bm\b|\bl\b|\bxl\b',
        'Color': r'\bcolor\b|\bcolour\b|\bshade\b|\btint\b|\bblack\b|\bwhite\b|\bblue\b|\bred\b|\bgreen\b',
        'Material': r'\bmaterial\b|\bfabric\b|\bleather\b|\bmetal\b|\bwood\b|\bcotton\b|\bplastic\b',
        'Style': r'\bstyle\b|\btype\b|\bdesign\b|\bpattern\b|\bfit\b|\bshape\b|\bslim\b|\bregular\b',
        'Quantity': r'\bquantity\b|\bpack\b|\bbundle\b|\bcount\b|\bnumber\b|\bpair\b',
        'Scent': r'\bscent\b|\bfragrance\b|\bflavor\b|\bflavour\b|\bperfume\b',
        'Hardware': r'\bhardware\b|\bbuckle\b|\bclasp\b|\bhook\b|\bchain\b|\brope\b',
        'Letter': r'\bletter\b|\bletra\b|\binitial\b|\bmonogram\b|\bname\b|\bfont\b'
    }
    
    doc = nlp(title_clean)
    for attr, pattern in regex_patterns.items():
        if re.search(pattern, title_clean) or any(token.lemma_ in pattern.split('|') for token in doc):
            primary_attr = attr
            break
    else:
        notes += "; Ambiguous title, requires manual review"
    
    if 'color' in title_clean and primary_attr != 'Color':
        secondary_attr = 'Color'
    elif 'size' in title_clean and primary_attr != 'Size':
        secondary_attr = 'Size'
    elif 'shape' in title_clean and primary_attr != 'Style':
        secondary_attr = 'Shape'
    
    return original_title, primary_attr, secondary_attr, group_id, notes

# Group similar titles with clustering
def group_similar_titles(titles, threshold=0.2):
    cleaned_titles = [clean_text(title) for title in titles]
    vectorizer = TfidfVectorizer().fit_transform(cleaned_titles)
    clustering = DBSCAN(eps=threshold, min_samples=2, metric='cosine').fit(vectorizer)
    
    title_to_group = {}
    for idx, label in enumerate(clustering.labels_):
        if label != -1:
            group_id = str(uuid.uuid4()) if label == 0 else f"group_{label}"
            title_to_group[titles[idx]] = group_id
        else:
            title_to_group[titles[idx]] = str(uuid.uuid4())
    
    return title_to_group

# Process full batch
def process_full_batch(input_file, pilot_size=1500, output_file='full_batch.xlsx'):
    logging.info("Milestone 2: Processing Full Batch...")
    df = pd.read_excel(input_file).iloc[pilot_size:]
    
    taxonomy, inverse_taxonomy = build_dynamic_taxonomy(df['variant_title'])
    unique_titles = df['variant_title'].dropna().unique()
    title_to_group = group_similar_titles(unique_titles)
    
    results = []
    for title in df['variant_title']:
        original, primary, secondary, temp_group_id, notes = normalize_variant_title(title, inverse_taxonomy, taxonomy)
        group_id = title_to_group.get(original, temp_group_id)
        results.append({
            'Original_Title': original,
            'Normalized_Title': primary,
            'Primary_Attribute': primary,
            'Secondary_Attribute': secondary,
            'Group_ID': group_id,
            'Notes': notes
        })
    
    result_df = pd.DataFrame(results)
    result_df.to_excel(output_file, index=False)
    logging.info(f"Full batch processed. Output saved to {output_file}")
    return result_df

# Main execution
if __name__ == "__main__":
    input_file = 'variant_data.xlsx'
    process_full_batch(input_file)