import pandas as pd
import re
import nltk
from fuzzywuzzy import fuzz
from langdetect import detect
from googletrans import Translator
from translate import Translator as FallbackTranslator
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

# Initialize translators
google_translator = Translator()
fallback_translator = FallbackTranslator(to_lang="en")

# Category-to-attribute mapping based on ideal table
CATEGORY_TO_ATTRIBUTES = {
    "Clothing Size": ["Size", "Fit Type", "Sleeve Length"],
    "Personalization": ["Letter", "Language"],
    "Accessories": ["Style", "Hardware Type", "Clasp Type", "Closure Type"],
    "Bedding": ["Material", "Pattern", "Finish"],
    "Jewelry": ["Style", "Jewelry Finish", "Material"],
    "Home Decor": ["Color", "Material", "Finish", "Style"],
    "Electronics": ["Voltage", "Connectivity", "Storage Size", "Model Year", "Compatibility"],
    "Beauty": ["Scent", "Shade", "Formula Type", "Skin Type"],
    "Pets": ["Flavor", "Scent", "Material"],
    "Health": ["Strength", "Flavor", "Delivery Type", "Target Use"],
    "Toys": ["Age Group", "Theme"],
    "Sports": ["Material", "Fit Type", "Style"],
    "Household": ["Pack Size", "Scent", "Usage Type"],
    "Food & Drink": ["Flavor", "Caffeine Level", "Variety"],
    "Arts & Crafts": ["Color", "Material", "Tip Type"],
    "Automotive": ["Material", "Voltage", "Compatibility"]
}

# Regex patterns for attribute extraction
REGEX_PATTERNS = {
    "Size": r"\b(size|s-m-l|xs|s|m|l|xl|xxl|inch|cm|ft|oz|diameter|length|height|width|back length|bead height|hoop diameter|strap size|neck size|waist size|drop length|cuff size|ring size|undies size|hxw|h x w|l x w x h|measured in thickness|taille)\b",
    "Color": r"\b(color|colour|shade|hue|black|white|red|blue|green|yellow|pink|grey|gray|olive|cream|mulberry|silver|charcoal|neon|pink|colorblock|stripes)\b",
    "Material": r"\b(material|fabric|cotton|leather|silk|wool|polyester|fill|cover|shell|stuffing|knit|sari|embroidered|pearl|charm|goose down)\b",
    "Style": r"\b(style|fit type|sleeve length|buckle|pattern|slim fit|regular fit|sienna|drape|hoodie|jogger|pant|shirt|skirt|top|single pearl|five pearl|charm|design|paw|shoe|wine glass)\b",
    "Finish": r"\b(finish|polish|matte|glossy|jewelry finish)\b",
    "Pack Size": r"\b(pack size|quantity|count|units per pack|pack type|single|pair|double)\b",
    "Hardware Type": r"\b(hardware|clasp type|closure type|buckle type)\b",
    "Scent": r"\b(scent|fragrance|aroma)\b",
    "Flavor": r"\b(flavor|taste|single origins)\b",
    "Voltage": r"\b(voltage|power|v|wattage)\b",
    "Connectivity": r"\b(connectivity|plug type|cable length|usb|bluetooth|wifi)\b",
    "Compatibility": r"\b(compatibility|model|year)\b",
    "Theme": r"\b(theme|edition|motif|sin print|tic-tac-toe|paw|shoe|wine glass)\b",
    "Age Group": r"\b(age group|age|kids|adult|senior)\b",
    "Strength": r"\b(strength|level|potency|firmness)\b",
    "Variety": r"\b(variety|type|variant|options)\b",
    "Weight Capacity": r"\b(weight capacity|weight|load)\b",
    "Language": r"\b(language|letra|letter)\b",
    "Letter": r"\b(letter|letra)\b",
    "Formula Type": r"\b(formula|skin type|cream|gel|lotion)\b",
    "Target Use": r"\b(target use|usage type|application)\b",
    "Caffeine Level": r"\b(caffeine level|caffeinated|decaf)\b",
    "Tip Type": r"\b(tip type|brush|pen)\b",
    "Delivery Type": r"\b(delivery type|spray|capsule|tablet)\b",
    "Storage Size": r"\b(storage size|capacity|gb|tb)\b",
    "Model Year": r"\b(model year|year|edition)\b",
    "Fit Type": r"\b(fit type|slim fit|regular fit)\b",
    "Sleeve Length": r"\b(sleeve length|short sleeve|long sleeve)\b",
    "Clasp Type": r"\b(clasp type|closure type)\b",
    "Jewelry Finish": r"\b(jewelry finish|polish)\b",
    "Pattern": r"\b(pattern|sin print|stripes|colorblock)\b",
    "Usage Type": r"\b(usage type|application)\b",
    "Shade": r"\b(shade|hue)\b",
    "Skin Type": r"\b(skin type|dry|oily|sensitive)\b",
    "Plug Type": r"\b(plug type|usb|type-c)\b",
    "Cable Length": r"\b(cable length|ft|m)\b",
    "Quantity": r"\b(quantity|count|single|pair|double)\b",
    "Closure Type": r"\b(closure type|clasp type|buckle type)\b",
    "Edition": r"\b(edition|version)\b",
    "Design": r"\b(design|charm|paw|shoe|wine glass)\b",
}

# Taxonomy definition with hierarchy
TAXONOMY = {
    "Size": ["size", "s-m-l", "strap size", "neck size", "waist size", "diameter", "length", "height", "width", "inch", "cm", "ft", "oz", "back length", "bead height", "hoop diameter", "drop length", "cuff size", "ring size", "undies size", "hxw", "h x w", "l x w x h", "taille"],
    "Color": ["color", "colour", "shade", "hue", "buckle color", "black", "white", "red", "blue", "green", "yellow", "pink", "grey", "gray", "olive", "cream", "mulberry", "silver", "charcoal", "neon", "pink", "colorblock", "stripes"],
    "Material": {
        "Fill": ["fill", "filling", "stuffing", "goose down"],
        "Cover": ["cover", "outer", "shell", "fabric", "leather"]
    },
    "Style": ["style", "fit type", "sleeve length", "buckle", "pattern", "slim fit", "regular fit", "sienna", "drape", "hoodie", "jogger", "pant", "shirt", "skirt", "top", "single pearl", "five pearl", "charm", "design"],
    "Finish": ["finish", "jewelry finish", "polish", "matte", "glossy"],
    "Pack Size": {
        "Units": ["quantity", "count", "single", "pair", "double"],
        "Pack Type": ["pack type"]
    },
    "Hardware Type": ["hardware", "clasp type", "closure type", "buckle type"],
    "Scent": ["scent", "fragrance", "aroma"],
    "Flavor": ["flavor", "taste", "single origins"],
    "Voltage": ["voltage", "power", "wattage"],
    "Connectivity": ["connectivity", "plug type", "cable length", "usb", "bluetooth", "wifi"],
    "Compatibility": ["compatibility", "model", "year"],
    "Theme": ["theme", "edition", "sin print", "tic-tac-toe", "paw", "shoe", "wine glass"],
    "Age Group": ["age group", "age", "kids", "adult", "senior"],
    "Strength": ["strength", "level", "potency", "firmness"],
    "Variety": ["variety", "type", "options"],
    "Language": ["language", "letra", "letter"],
    "Letter": ["letter", "letra"],
    "Formula Type": ["formula", "skin type", "cream", "gel", "lotion"],
    "Target Use": ["target use", "usage type", "application"],
    "Caffeine Level": ["caffeine level", "caffeinated", "decaf"],
    "Tip Type": ["tip type", "brush", "pen"],
    "Delivery Type": ["delivery type", "spray", "capsule", "tablet"],
    "Storage Size": ["storage size", "capacity", "gb", "tb"],
    "Model Year": ["model year", "year"],
    "Fit Type": ["fit type", "slim fit", "regular fit"],
    "Sleeve Length": ["sleeve length", "short sleeve", "long sleeve"],
    "Clasp Type": ["clasp type", "closure type"],
    "Jewelry Finish": ["jewelry finish", "polish"],
    "Pattern": ["pattern", "sin print", "stripes", "colorblock"],
    "Shade": ["shade", "hue"],
    "Skin Type": ["skin type", "dry", "oily", "sensitive"],
    "Plug Type": ["plug type", "usb", "type-c"],
    "Cable Length": ["cable length", "ft", "m"],
    "Quantity": ["quantity", "count", "single", "pair", "double"],
    "Closure Type": ["closure type", "clasp type", "buckle type"],
    "Design": ["design", "charm", "paw", "shoe", "wine glass"],
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
    """Detect language of the title using langdetect."""
    try:
        return detect(title)
    except Exception as e:
        logging.warning(f"Language detection failed for '{title}': {e}")
        return 'en'

def translate_to_english(title: str, src_lang: str) -> Tuple[str, str]:
    """Translate title to English using googletrans, with translate as fallback."""
    if src_lang == 'en' or not title:
        return title, ""
    try:
        result = google_translator.translate(title, src=src_lang, dest='en').text
        notes = f"Translated from {src_lang.capitalize()}"
        logging.info(f"Translated '{title}' from {src_lang} to '{result}'")
        return result, notes
    except Exception as e:
        logging.warning(f"Googletrans failed for '{title}': {e}. Trying fallback translator.")
        try:
            result = fallback_translator.translate(title)
            notes = f"Translated from {src_lang.capitalize()} (fallback)"
            logging.info(f"Translated '{title}' from {src_lang} to '{result}' (fallback)")
            return result, notes
        except Exception as e:
            logging.warning(f"Fallback translation failed for '{title}': {e}")
            return title, "Translation failed"

def normalize_text(text: str) -> str:
    """Clean text for matching."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def extract_normalized_value(title: str, primary_attr: str, secondary_attr: str) -> str:
    """Extract normalized value from title."""
    title_lower = title.lower()
    normalized = ""
    
    if primary_attr == "Size" and secondary_attr:
        normalized = secondary_attr
    elif primary_attr == "Color":
        color_match = re.search(r'\b(black|white|red|blue|green|yellow|pink|grey|gray|olive|cream|mulberry|silver|charcoal|neon)\b', title_lower)
        if color_match:
            normalized = color_match.group(0).capitalize()
    elif primary_attr == "Material" and secondary_attr == "Fill":
        material_match = re.search(r'\b(cotton|goose down|leather|polyester)\b', title_lower)
        if material_match:
            normalized = material_match.group(0).capitalize()
    elif primary_attr == "Style" and secondary_attr == "Design":
        design_match = re.search(r'\b(paw|shoe|wine glass|charm)\b', title_lower)
        if design_match:
            normalized = f"{design_match.group(0).capitalize()} Charm"
    elif primary_attr in ["Letter", "Language"]:
        normalized = "Letter"
    elif primary_attr == "Size":
        normalized = "Size"
    
    return normalized if normalized else title

def infer_category(primary_attr: str, title: str) -> Tuple[str, str]:
    """Infer category based on PrimaryAttribute and title context."""
    if primary_attr == "Ambiguous":
        return "Unknown", "Category unknown due to ambiguous attribute"
    
    possible_categories = [cat for cat, attrs in CATEGORY_TO_ATTRIBUTES.items() if primary_attr in attrs]
    
    if not possible_categories:
        return "Unknown", f"Category unknown for attribute {primary_attr}"
    
    if len(possible_categories) == 1:
        return possible_categories[0], f"Category inferred as {possible_categories[0]}"
    
    title_lower = title.lower()
    if primary_attr in ["Size", "Color", "Material", "Style"]:
        if any(keyword in title_lower for keyword in ["knit", "sari", "shirt", "skirt", "top", "jogger", "hoodie"]):
            return "Clothing Size", "Category inferred as Clothing Size based on title context"
        if any(keyword in title_lower for keyword in ["charm", "pearl", "necklace", "bracelet", "earring"]):
            return "Jewelry", "Category inferred as Jewelry based on title context"
        if any(keyword in title_lower for keyword in ["fill", "cover", "bedding", "pillow"]):
            return "Bedding", "Category inferred as Bedding based on title context"
        if any(keyword in title_lower for keyword in ["buckle", "clasp", "hardware"]):
            return "Accessories", "Category inferred as Accessories based on title context"
        if any(keyword in title_lower for keyword in ["decor", "furniture"]):
            return "Home Decor", "Category inferred as Home Decor based on title context"
        return "Clothing Size", "Category inferred as Clothing Size (default for generic attribute)"
    
    if primary_attr in ["Letter", "Language"]:
        return "Personalization", "Category inferred as Personalization"
    if primary_attr in ["Scent", "Shade", "Formula Type", "Skin Type"]:
        return "Beauty", "Category inferred as Beauty"
    if primary_attr in ["Voltage", "Connectivity", "Storage Size", "Model Year"]:
        return "Electronics", "Category inferred as Electronics"
    
    return possible_categories[0], f"Category inferred as {possible_categories[0]} (first match)"

def match_to_taxonomy(title: str, taxonomy: Dict, regex_patterns: Dict) -> Tuple[str, str, str, str]:
    """Match title to taxonomy, extracting primary and secondary attributes."""
    normalized_title = normalize_text(title)
    tokens = nltk.word_tokenize(normalized_title)
    
    primary_attr = ""
    secondary_attr = ""
    hierarchy_level1 = ""
    hierarchy_level2 = ""
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
        hierarchy_level1 = best_match
        if best_match in ["Fill", "Cover"]:
            primary_attr = "Material"
            hierarchy_level1 = "Material"
            secondary_attr = best_match
            hierarchy_level2 = best_match
        elif best_match in ["Units", "Pack Type"]:
            primary_attr = "Pack Size"
            hierarchy_level1 = "Pack Size"
            secondary_attr = best_match
            hierarchy_level2 = best_match
    
    if len(matched_attrs) > 1:
        primary_attr = matched_attrs[0]
        secondary_attr = matched_attrs[1]
        hierarchy_level1 = primary_attr
        hierarchy_level2 = secondary_attr
        notes = "Multiple attributes detected"
    elif len(matched_attrs) == 1 and not primary_attr:
        primary_attr = matched_attrs[0]
        hierarchy_level1 = primary_attr
    
    if not primary_attr:
        primary_attr = "Ambiguous"
        notes = "Ambiguous - manual review"
    
    return primary_attr, secondary_attr, hierarchy_level1, hierarchy_level2, notes

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

def process_title(row, row_id: int, taxonomy, regex_patterns):
    """Process a single title (used for parallel processing)."""
    title = str(row['variant_title'])
    
    preprocessed_title = preprocess_title(title)
    lang = detect_language(preprocessed_title)
    lang_name = {"en": "English", "es": "Spanish", "fr": "French"}.get(lang, "English")
    title_english, translation_notes = translate_to_english(preprocessed_title, lang)
    primary_attr, secondary_attr, hierarchy_level1, hierarchy_level2, notes = match_to_taxonomy(title_english, taxonomy, regex_patterns)
    category, category_notes = infer_category(primary_attr, title_english)
    normalized_value = extract_normalized_value(title_english, primary_attr, secondary_attr)
    group_id = f"{row_id:03d}"
    ambiguity_flag = "Yes" if primary_attr == "Ambiguous" or (secondary_attr and primary_attr not in ["Material", "Pack Size"]) else "No"
    final_notes = f"{translation_notes}; {notes}; {category_notes}" if translation_notes else f"{notes}; {category_notes}"
    
    return {
        "Row ID": row_id,
        "Original Title": title,
        "Primary Attribute": primary_attr,
        "Secondary Attribute": secondary_attr,
        "Group ID": group_id,
        "Normalized Value": normalized_value,
        "Category": category,
        "Hierarchy Level 1": hierarchy_level1,
        "Hierarchy Level 2": hierarchy_level2,
        "Language": lang_name,
        "Ambiguity Flag": ambiguity_flag,
        "Notes": final_notes
    }

def process_full_batch(input_file: str, output_file: str, taxonomy_file: str):
    """Process the full batch of 11,500+ variant titles (Milestone 2)."""
    logging.info("Starting Milestone 2: Full Batch Processing")
    
    if not os.path.exists(input_file):
        logging.error(f"Input file {input_file} not found")
        raise FileNotFoundError(f"Input file {input_file} not found")
    
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
            delayed(process_title)(row, row_id=start + idx + 1, taxonomy=taxonomy, regex_patterns=REGEX_PATTERNS)
            for idx, row in batch.iterrows()
        )
        results.extend(batch_results)
        logging.info(f"Processed batch {start // batch_size + 1}/{len(df) // batch_size + 1}")
    
    result_df = pd.DataFrame(results)
    
    ambiguous_titles = result_df[result_df['Primary Attribute'] == 'Ambiguous']['Original Title'].tolist()
    if ambiguous_titles:
        clusters = cluster_ambiguous_titles(ambiguous_titles)
        for cluster_id, titles in clusters.items():
            for title in titles:
                idx = result_df[result_df['Original Title'] == title].index
                result_df.loc[idx, 'Notes'] = f"Clustered as {cluster_id} for review; {result_df.loc[idx, 'Notes']}"
    
    try:
        result_df.to_excel(output_file, index=False, engine='openpyxl')
        logging.info(f"Full batch output saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving output: {e}")
        raise
    
    ambiguous_count = len(result_df[result_df['Primary Attribute'] == 'Ambiguous'])
    logging.info(f"Processed {len(result_df)} titles. {ambiguous_count} marked as ambiguous.")
    return result_df

if __name__ == "__main__":
    input_file = "variant_data.xlsx"
    output_file = "full_normalized_titles.xlsx"
    taxonomy_file = "taxonomy.json"
    process_full_batch(input_file, output_file, taxonomy_file)