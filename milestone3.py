import pandas as pd
import json
import logging
from typing import Dict, List
import warnings
import os
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Taxonomy definition (for fallback if taxonomy.json is missing)
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

def load_data(pilot_file: str, full_file: str) -> pd.DataFrame:
    """Load and combine pilot and full batch outputs."""
    try:
        pilot_df = pd.read_excel(pilot_file, engine='openpyxl')
    except FileNotFoundError:
        logging.warning(f"Pilot file not found. Creating dummy {pilot_file}.")
        pilot_df = pd.DataFrame([{
            "Row ID": 1,
            "Original Title": "Sample Size",
            "Primary Attribute": "Size",
            "Secondary Attribute": "",
            "Group ID": "001",
            "Normalized Value": "Size",
            "Category": "Clothing Size",
            "Hierarchy Level 1": "Size",
            "Hierarchy Level 2": "",
            "Language": "English",
            "Ambiguity Flag": "No",
            "Notes": "Dummy data; Category inferred as Clothing Size"
        }])
        pilot_df.to_excel(pilot_file, index=False, engine='openpyxl')
    
    try:
        full_df = pd.read_excel(full_file, engine='openpyxl')
        combined_df = pd.concat([pilot_df, full_df], ignore_index=True)
        logging.info(f"Loaded {len(pilot_df)} pilot rows and {len(full_df)} full batch rows")
        return combined_df
    except Exception as e:
        logging.error(f"Error loading files: {e}")
        raise

def load_taxonomy(taxonomy_file: str) -> Dict:
    """Load taxonomy from JSON file or use TAXONOMY as fallback."""
    try:
        with open(taxonomy_file, 'r') as f:
            taxonomy = json.load(f)
        logging.info(f"Loaded taxonomy from {taxonomy_file}")
        return taxonomy
    except FileNotFoundError:
        logging.warning(f"Taxonomy file not found. Using default TAXONOMY.")
        return TAXONOMY
    except Exception as e:
        logging.error(f"Error loading taxonomy: {e}")
        raise

def review_ambiguous_titles(df: pd.DataFrame, taxonomy: Dict, manual_mappings: Dict = None) -> pd.DataFrame:
    """Review and resolve ambiguous titles."""
    if manual_mappings is None:
        manual_mappings = {}
    
    ambiguous_df = df[df['Primary Attribute'] == 'Ambiguous'].copy()
    resolved_rows = []
    
    for _, row in ambiguous_df.iterrows():
        title = row['Original Title']
        group_id = row['Group ID']
        
        if title in manual_mappings:
            primary_attr, secondary_attr, category, normalized_value = manual_mappings[title]
            notes = f"Resolved via manual mapping; Category inferred as {category}"
            ambiguity_flag = "No"
        else:
            primary_attr = "Ambiguous"
            secondary_attr = ""
            category = "Unknown"
            normalized_value = title
            notes = f"Clustered as {group_id} - requires manual review; Category unknown"
            ambiguity_flag = "Yes"
        
        resolved_rows.append({
            "Row ID": row['Row ID'],
            "Original Title": row['Original Title'],
            "Primary Attribute": primary_attr,
            "Secondary Attribute": secondary_attr,
            "Group ID": group_id,
            "Normalized Value": normalized_value,
            "Category": category,
            "Hierarchy Level 1": primary_attr,
            "Hierarchy Level 2": secondary_attr,
            "Language": row['Language'],
            "Ambiguity Flag": ambiguity_flag,
            "Notes": notes
        })
    
    resolved_df = pd.DataFrame(resolved_rows)
    non_ambiguous_df = df[df['Primary Attribute'] != 'Ambiguous']
    final_df = pd.concat([non_ambiguous_df, resolved_df], ignore_index=True)
    return final_df

def finalize_taxonomy(taxonomy: Dict, df: pd.DataFrame, taxonomy_file: str):
    """Finalize taxonomy by incorporating new attributes."""
    new_attributes = set(df[df['Primary Attribute'] != 'Ambiguous']['Primary Attribute']) - set(taxonomy.keys())
    
    for attr in new_attributes:
        if attr not in taxonomy:
            taxonomy[attr] = []
    
    try:
        with open(taxonomy_file, 'w') as f:
            json.dump(taxonomy, f, indent=2)
        logging.info(f"Updated taxonomy saved to {taxonomy_file}")
    except Exception as e:
        logging.error(f"Error saving taxonomy: {e}")
        raise

def generate_summary(df: pd.DataFrame, summary_file: str):
    """Generate summary of edge cases and assumptions."""
    ambiguous_count = len(df[df['Ambiguity Flag'] == 'Yes'])
    clusters = df[df['Primary Attribute'] == 'Ambiguous']['Group ID'].value_counts()
    
    summary = {
        "Total Titles Processed": len(df),
        "Ambiguous Titles": ambiguous_count,
        "Ambiguous Clusters": clusters.to_dict(),
        "Assumptions": [
            "Titles not matching taxonomy were marked as Ambiguous and clustered for review.",
            "Categories inferred based on Primary Attribute and title context.",
            "Generic attributes (e.g., Size) defaulted to Clothing Size if no clear context.",
            "Normalized values extracted based on attribute-specific rules.",
            "Long dashes were converted to hyphens during preprocessing.",
            "Titles were title-cased for consistency."
        ],
        "Edge Cases": [
            "Multilingual titles (e.g., 'Letra') were translated to English.",
            "Titles with multiple attributes (e.g., 'Size - Cream/Mulberry') flagged as ambiguous.",
            "Titles with ambiguous context (e.g., 'Buckle') were clustered for manual review.",
            "Titles with measurements (e.g., 'Size (H x W)') were mapped to Size."
        ]
    }
    
    try:
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logging.info(f"Summary saved to {summary_file}")
    except Exception as e:
        logging.error(f"Error saving summary: {e}")
        raise

def process_qa_review(pilot_file: str, full_file: str, output_file: str, taxonomy_file: str, summary_file: str):
    """Process Milestone 3: Review and QA."""
    logging.info("Starting Milestone 3: Review & QA")
    
    df = load_data(pilot_file, full_file)
    taxonomy = load_taxonomy(taxonomy_file)
    
    manual_mappings = {
        # Placeholder: Add client-provided mappings, e.g., "Some Title": ("Size", "", "Clothing Size", "Size")
    }
    df = review_ambiguous_titles(df, taxonomy, manual_mappings)
    finalize_taxonomy(taxonomy, df, taxonomy_file)
    
    try:
        df.to_excel(output_file, index=False, engine='openpyxl')
        logging.info(f"Final cleaned output saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving output: {e}")
        raise
    
    generate_summary(df, summary_file)

if __name__ == "__main__":
    pilot_file = "pilot_normalized_titles.xlsx"
    full_file = "full_normalized_titles.xlsx"
    output_file = "normalized_variants.xlsx"
    taxonomy_file = "taxonomy.json"
    summary_file = "qa_summary.json"
    process_qa_review(pilot_file, full_file, output_file, taxonomy_file, summary_file)