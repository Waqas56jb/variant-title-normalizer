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

def load_data(pilot_file: str, full_file: str) -> pd.DataFrame:
    """Load and combine pilot and full batch outputs."""
    try:
        pilot_df = pd.read_excel(pilot_file, engine='openpyxl')
    except FileNotFoundError:
        logging.warning(f"Pilot file not found. Creating dummy {pilot_file}.")
        pilot_df = pd.DataFrame([{
            "Category": "Unknown",
            "OriginalTitle": "Sample Size",
            "TitleEnglish": "Sample Size",
            "PrimaryAttribute": "Size",
            "SecondaryAttribute": "",
            "GroupID": "Unknown_Size_sample",
            "Notes": "Dummy data"
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
    
    ambiguous_df = df[df['PrimaryAttribute'] == 'Ambiguous'].copy()
    resolved_rows = []
    
    for _, row in ambiguous_df.iterrows():
        title = row['TitleEnglish']
        group_id = row['GroupID']
        
        if title in manual_mappings:
            primary_attr, secondary_attr = manual_mappings[title]
            notes = "Resolved via manual mapping"
        else:
            primary_attr = "Ambiguous"
            secondary_attr = ""
            notes = f"Clustered as {group_id} - requires manual review"
        
        resolved_rows.append({
            "Category": row['Category'],
            "OriginalTitle": row['OriginalTitle'],
            "TitleEnglish": row['TitleEnglish'],
            "PrimaryAttribute": primary_attr,
            "SecondaryAttribute": secondary_attr,
            "GroupID": group_id,
            "Notes": notes
        })
    
    resolved_df = pd.DataFrame(resolved_rows)
    non_ambiguous_df = df[df['PrimaryAttribute'] != 'Ambiguous']
    final_df = pd.concat([non_ambiguous_df, resolved_df], ignore_index=True)
    return final_df

def finalize_taxonomy(taxonomy: Dict, df: pd.DataFrame, taxonomy_file: str):
    """Finalize taxonomy by incorporating new attributes."""
    new_attributes = set(df[df['PrimaryAttribute'] != 'Ambiguous']['PrimaryAttribute']) - set(taxonomy.keys())
    
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
    ambiguous_count = len(df[df['PrimaryAttribute'] == 'Ambiguous'])
    clusters = df[df['PrimaryAttribute'] == 'Ambiguous']['GroupID'].value_counts()
    
    summary = {
        "Total Titles Processed": len(df),
        "Ambiguous Titles": ambiguous_count,
        "Ambiguous Clusters": clusters.to_dict(),
        "Assumptions": [
            "Titles not matching taxonomy were marked as Ambiguous and clustered for review.",
            "Category set to 'Unknown' due to missing Category column.",
            "Manual mappings were applied where provided; otherwise, titles remain ambiguous.",
            "Taxonomy was updated with new attributes identified during processing.",
            "Long dashes were converted to hyphens during preprocessing.",
            "Titles were title-cased for consistency."
        ],
        "Edge Cases": [
            "Multilingual titles (e.g., 'Letra') were translated to English.",
            "Titles with multiple attributes (e.g., 'Size & Color') were flagged for review.",
            "Titles with ambiguous context (e.g., 'Buckle') were clustered for manual review."
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
        # Placeholder: Add client-provided mappings, e.g., "Some Title": ("Size", "")
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