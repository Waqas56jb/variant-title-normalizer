import pandas as pd
import re
import nltk
from fuzzywuzzy import fuzz
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import logging
import uuid
import json
from joblib import Parallel, delayed
from typing import Dict, List, Tuple
import warnings
import os
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('punkt', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize transformer pipelines (CPU for local execution)
language_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection", device="cpu")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en", device="cpu")

# Regex patterns for attribute extraction
REGEX_PATTERNS = {
    "Size": r"\b(size|s-m-l|xs|s|m|l|xl|xxl|inch|cm|ft|diameter|length|height|width|oz|back length|bead height|hoop diameter)\b",
    "Color": r"\b(color|colour|shade|hue|black|white|red|blue|green|yellow|pink|grey|gray|olive|cream|mulberry)\b",
    "Material": r"\b(material|fabric|cotton|leather|silk|wool|polyester|fill|cover|shell|stuffing)\b",
    "Style": r"\b(style|fit type|sleeve length|buckle|pattern|slim fit|regular fit|knit|sari|embroidered)\b",
    "Finish": r"\b(finish|polish|matte|glossy|jewelry finish)\b",
    "Pack Size": r"\b(pack size|quantity|count|units per pack|pack type)\b",
    "Hardware Type": r"\b(hardware|clasp type|closure type|buckle type)\b",
    "Scent": r"\b(scent|fragrance|aroma)\b",
    "Flavor": r"\b(flavor|taste)\b",
    "Voltage": r"\b(voltage|power|v|wattage)\b",
    "Connectivity": r"\b(connectivity|plug type|cable length|usb|bluetooth|wifi)\b",
    "Compatibility": r"\b(compatibility|model|year)\b",
    "Theme": r"\b(theme|edition|motif)\b",
    "Age Group": r"\b(age group|age|kids|adult|senior)\b",
    "Strength": r"\b(strength|level|potency)\b",
    "Variety": r"\b(variety|type|variant)\b",
    "Weight Capacity": r"\b(weight capacity|weight|load)\b",
    "Language": r"\b(language|letra|letter)\b",
    "Formula Type": r"\b(formula|skin type|cream|gel|lotion)\b",
    "Target Use": r"\b(target use|usage type|application)\b",
    "Platform": r"\b(platform|console|device)\b",
    "Caffeine Level": r"\b(caffeine level|caffeinated|decaf)\b",
    "Tip Type": r"\b(tip type|brush|pen)\b",
    "Delivery Type": r"\b(delivery type|spray|capsule|tablet)\b",
    "Storage Size": r"\b(storage size|capacity|gb|tb)\b",
    "Model Year": r"\b(model year|year|edition)\b",
}

# Taxonomy definition with hierarchy
TAXONOMY = {
    "Size": ["size", "s-m-l", "strap size", "neck size", "waist size", "diameter", "length", "height", "width", "inch", "cm", "ft"],
    "Color": ["color", "colour", "shade", "hue", "buckle color"],
    "Material": {
        "Fill Material": ["fill", "filling", "stuffing"],
        "Cover Material": ["cover", "outer", "shell", "fabric"]
    },
    "Style": ["style", "fit type", "sleeve length", "buckle", "pattern"],
    "Finish": ["finish", "jewelry finish"],
    "Pack Size": {
        "Units per Pack": ["quantity", "count"],
        "Pack Type": ["pack type"]
    },
    "Hardware Type": ["hardware", "clasp type", "closure type"],
    "Scent": ["scent", "fragrance"],
    "Flavor": ["flavor", "taste"],
    "Voltage": ["voltage", "power"],
    "Connectivity": ["connectivity", "plug type", "cable length"],
    "Compatibility": ["compatibility", "model", "year"],
    "Theme": ["theme", "edition"],
    "Age Group": ["age group", "age"],
    "Strength": ["strength", "level"],
    "Variety": ["variety", "type"],
    "Weight Capacity": ["weight capacity", "weight"],
    "Language": ["language", "letra", "letter"],
    "Formula Type": ["formula", "skin type"],
    "Target Use": ["target use", "usage type"],
    "Platform": ["platform"],
    "Caffeine Level": ["caffeine level"],
    "Tip Type": ["tip type"],
    "Delivery Type": ["delivery type"],
    "Storage Size": ["storage size", "capacity"],
    "Model Year": ["model year"],
}

def preprocess_title(title: str) -> str:
    """Preprocess title: trim whitespace, normalize dashes, title-case."""
    if not isinstance(title, str):
        return ""
    title = re.sub(r'\s+', ' ', title.strip())
    title = re.sub(r'[–—]', '-', title)
    title = ' '.join(word.capitalize() for word in title.split())
    return title

def detect_language(title: str) -> str:
    """Detect language of the title."""
    try:
        result = language_detector(title, truncation=True, max_length=512)
        return result[0]['label']
    except Exception as e:
        logging.warning(f"Language detection failed for '{title}': {e}")
        return 'en'

def translate_to_english(title: str, src_lang: str) -> Tuple[str, str]:
    """Translate title to English if not already in English."""
    if src_lang == 'en' or not title:
        return title, ""
    try:
        result = translator(title, src_lang=src_lang, tgt_lang='en')[0]['translation_text']
        notes = f"Translated from {src_lang.capitalize()}"
        logging.info(f"Translated '{title}' from {src_lang} to '{result}'")
        return result, notes
    except Exception as e:
        logging.warning(f"Translation failed for '{title}': {e}")
        return title, "Translation failed"

def normalize_text(text: str) -> str:
    """Clean text for matching."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def match_to_taxonomy(title: str, taxonomy: Dict, regex_patterns: Dict) -> Tuple[str, str, str]:
    """Match title to taxonomy, extracting primary and secondary attributes."""
    normalized_title = normalize_text(title)
    tokens = nltk.word_tokenize(normalized_title)
    
    primary_attr = ""
    secondary_attr = ""
    notes = "Mapped by fuzzy match"
    
    matched_attrs = []
    for attr, pattern in regex_patterns.items():
        if re.search(pattern, normalized_title, re.IGNORECASE):
            matched_attrs.append(attr)
    
    best_match = None
    best_score = 0
    def search_taxonomy(tax: Dict | List, parent: str = ""):
        nonlocal best_match, best_score
        if isinstance(tax, list):
            for term in tax:
                score = fuzz.partial_ratio(normalized_title, term)
                if score > best_score and score >= 80:
                    best_score = score
                    best_match = parent
        elif isinstance(tax, dict):
            for key, value in tax.items():
                score = fuzz.partial_ratio(normalized_title, key.lower())
                if score > best_score and score >= 80:
                    best_score = score
                    best_match = key
                search_taxonomy(value, key)
    
    search_taxonomy(taxonomy)
    
    if best_match:
        primary_attr = best_match
        if best_match in ["Fill Material", "Cover Material"]:
            primary_attr = "Material"
        elif best_match in ["Units per Pack", "Pack Type"]:
            primary_attr = "Pack Size"
    
    if len(matched_attrs) > 1:
        primary_attr = matched_attrs[0]
        secondary_attr = matched_attrs[1]
        notes = "Multiple attributes detected"
    elif len(matched_attrs) == 1 and not primary_attr:
        primary_attr = matched_attrs[0]
    
    if not primary_attr:
        primary_attr = "Ambiguous"
        notes = "Ambiguous - manual review"
    
    return primary_attr, secondary_attr, notes

def cluster_ambiguous_titles(titles: List[str], eps: float = 0.5, min_samples: int = 2) -> Dict[str, List[str]]:
    """Cluster ambiguous titles using DBSCAN."""
    if not titles:
        return {}
    
    try:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(titles)
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(X)
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:
                cluster_id = f"Outlier_{uuid.uuid4().hex[:8]}"
            else:
                cluster_id = f"Cluster_{label}"
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(titles[idx])
        return clusters
    except Exception as e:
        logging.error(f"Clustering failed: {e}")
        return {f"Outlier_{uuid.uuid4().hex[:8]}": titles}

def process_title(row, taxonomy, regex_patterns):
    """Process a single title (used for parallel processing)."""
    title = str(row['variant_title'])
    category = "Unknown"
    
    preprocessed_title = preprocess_title(title)
    lang = detect_language(preprocessed_title)
    title_english, translation_notes = translate_to_english(preprocessed_title, lang)
    primary_attr, secondary_attr, notes = match_to_taxonomy(title_english, taxonomy, regex_patterns)
    group_id = f"{category}_{primary_attr}_{uuid.uuid4().hex[:8]}" if primary_attr != "Ambiguous" else f"Ambiguous_{uuid.uuid4().hex[:8]}"
    final_notes = translation_notes + "; " + notes if translation_notes else notes
    
    return {
        "Category": category,
        "OriginalTitle": title,
        "TitleEnglish": title_english,
        "PrimaryAttribute": primary_attr,
        "SecondaryAttribute": secondary_attr,
        "GroupID": group_id,
        "Notes": final_notes
    }

def process_full_batch(input_file: str, output_file: str, taxonomy_file: str):
    """Process the full batch of 11,000+ variant titles (Milestone 2)."""
    logging.info("Starting Milestone 2: Full Batch Processing")
    
    try:
        with open(taxonomy_file, 'r') as f:
            taxonomy = json.load(f)
        logging.info(f"Loaded taxonomy from {taxonomy_file}")
    except FileNotFoundError:
        logging.warning(f"Taxonomy file not found. Creating {taxonomy_file} from TAXONOMY.")
        taxonomy = TAXONOMY
        with open(taxonomy_file, 'w') as f:
            json.dump(taxonomy, f, indent=2)
    
    try:
        df = pd.read_excel(input_file, engine='openpyxl')
        if 'variant_title' not in df.columns:
            raise ValueError("Input file must contain column: variant_title")
        logging.info(f"Loaded {len(df)} rows from {input_file}")
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        raise
    
    batch_size = 1000
    results = []
    for start in range(0, len(df), batch_size):
        batch = df[start:start + batch_size]
        batch_results = Parallel(n_jobs=4)(
            delayed(process_title)(row, taxonomy, REGEX_PATTERNS) for _, row in batch.iterrows()
        )
        results.extend(batch_results)
        logging.info(f"Processed batch {start // batch_size + 1}/{len(df) // batch_size + 1}")
    
    result_df = pd.DataFrame(results)
    
    ambiguous_titles = result_df[result_df['PrimaryAttribute'] == 'Ambiguous']['TitleEnglish'].tolist()
    if ambiguous_titles:
        clusters = cluster_ambiguous_titles(ambiguous_titles)
        for cluster_id, titles in clusters.items():
            for title in titles:
                idx = result_df[result_df['TitleEnglish'] == title].index
                result_df.loc[idx, 'GroupID'] = cluster_id
                result_df.loc[idx, 'Notes'] = f"Clustered as {cluster_id} for review; {result_df.loc[idx, 'Notes']}"
    
    try:
        result_df.to_excel(output_file, index=False, engine='openpyxl')
        logging.info(f"Full batch output saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving output: {e}")
        raise
    
    ambiguous_count = len(result_df[result_df['PrimaryAttribute'] == 'Ambiguous'])
    logging.info(f"Processed {len(result_df)} titles. {ambiguous_count} marked as ambiguous.")
    return result_df

if __name__ == "__main__":
    input_file = "variant_data.xlsx"
    output_file = "full_normalized_titles.xlsx"
    taxonomy_file = "taxonomy.json"
    process_full_batch(input_file, output_file, taxonomy_file)