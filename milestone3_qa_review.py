import pandas as pd
import re
import logging
import spacy
import uuid
from fuzzywuzzy import fuzz

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize NLP
nlp = spacy.load('en_core_web_sm')

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

# Attribute hierarchy
ATTRIBUTE_HIERARCHY = {
    'Material': ['Fill Material', 'Cover Material', 'Fabric', 'Leather', 'Metal'],
    'Size': ['Length', 'Width', 'Height', 'Diameter'],
    'Style': ['Design', 'Pattern', 'Shape', 'Finish']
}

# Build inverse taxonomy
def build_inverse_taxonomy():
    return {v.lower(): k for k, vs in INITIAL_CATEGORIES.items() for v in vs}

# Enhanced QA rules
def apply_qa_rules(title, primary_attr, secondary_attr, hierarchical_attr, notes, inverse_taxonomy):
    title_clean = str(title).lower()
    doc = nlp(title_clean)
    
    if re.search(r'\bsize\b|\blength\b|\bwidth\b|\bheight\b|\bgrandeur\b|\btalla\b', title_clean):
        return 'Size', secondary_attr if secondary_attr else None, 'Size' if not hierarchical_attr else hierarchical_attr, 'Refined to Size'
    elif re.search(r'\bcolor\b|\bcolour\b|\bcouleur\b|\bshade\b|\btint\b', title_clean):
        return 'Color', secondary_attr if secondary_attr else None, None, 'Refined to Color'
    elif any(token.lemma_ in ['style', 'type', 'design', 'finish'] for token in doc):
        return 'Style', secondary_attr if secondary_attr else None, 'Style' if not hierarchical_attr else hierarchical_attr, 'Refined to Style'
    elif re.search(r'\bmaterial\b|\bfabric\b|\bleather\b|\bmetal\b', title_clean):
        return 'Material', secondary_attr if secondary_attr else None, 'Material' if not hierarchical_attr else hierarchical_attr, 'Refined to Material'
    
    best_score = 0
    best_match = None
    for taxonomy_title in inverse_taxonomy.keys():
        score = fuzz.token_sort_ratio(title_clean, taxonomy_title)
        if score > best_score and score > 80:
            best_score = score
            best_match = taxonomy_title
    
    if best_match:
        primary_attr = inverse_taxonomy[best_match]
        notes += f"; Refined via fuzzy match: {best_match} (score: {best_score})"
        for parent, children in ATTRIBUTE_HIERARCHY.items():
            if primary_attr in children:
                hierarchical_attr = parent
                break
    
    return primary_attr, secondary_attr, hierarchical_attr, notes

# Refine taxonomy
def refine_taxonomy(df, inverse_taxonomy):
    flagged = df[df['Notes'].str.contains('Unmatched|Missing|Unknown', na=False, case=False)]
    refined_rows = []
    
    for _, row in flagged.iterrows():
        title = row['Original_Title']
        primary, secondary, hierarchical, new_notes = apply_qa_rules(
            title, row['Primary_Attribute'], row['Secondary_Attribute'], 
            row['Hierarchical_Attribute'], row['Notes'], inverse_taxonomy
        )
        refined_rows.append({
            'Original_Title': title,
            'Translated_Title': row['Translated_Title'],
            'Normalized_Title': row['Normalized_Title'],
            'Primary_Attribute': primary,
            'Secondary_Attribute': secondary,
            'Hierarchical_Attribute': hierarchical,
            'Group_ID': row['Group_ID'],
            'Notes': new_notes
        })
    
    refined_df = pd.DataFrame(refined_rows)
    df.update(refined_df)
    df['Notes'] = df['Notes'].fillna('')
    return df

def review_and_qa(input_file, output_file):
    logging.info("Milestone 3: Performing QA and Review...")
    
    # Load data
    df = pd.read_excel(input_file)
    inverse_taxonomy = build_inverse_taxonomy()
    
    # Refine taxonomy
    df = refine_taxonomy(df, inverse_taxonomy)
    
    # Generate summary
    edge_cases = df[df['Notes'].str.contains('manual review|Unknown', na=False, case=False)]
    summary = {
        'Total Titles': len(df),
        'Flagged Titles': len(edge_cases),
        'Unique Groups': df['Group_ID'].nunique(),
        'Unresolved Attributes': edge_cases['Primary_Attribute'].value_counts().to_dict()
    }
    
    # Save output
    df.to_excel(output_file, index=False)
    with open("output/summary.txt", "w") as f:
        f.write(str(summary))
    logging.info(f"QA complete. Output saved to {output_file}")
    return df

if __name__ == "__main__":
    review_and_qa("output/milestone2_full_output.xlsx", "output/milestone3_final_output.xlsx")