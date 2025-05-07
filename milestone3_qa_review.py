import pandas as pd
import re
import logging
import spacy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize NLP
nlp = spacy.load('en_core_web_sm')

# Enhanced QA rules
def apply_qa_rules(title, primary_attr, secondary_attr, notes):
    title_clean = title.lower()
    doc = nlp(title_clean)
    
    if 'co2' in title_clean:
        return 'Other', None, 'Assigned to Other (CO2-related, manual review needed)'
    elif 'shape' in title_clean or any(token.lemma_ == 'shape' for token in doc):
        return 'Style', 'Shape', 'Assigned to Style/Shape'
    elif 'buckle' in title_clean or 'clasp' in title_clean:
        return 'Hardware', 'Style', 'Assigned to Hardware/Style'
    elif 'scent' in title_clean or 'fragrance' in title_clean:
        return 'Scent', None, 'Assigned to Scent'
    elif re.search(r'\bsize\b|\bxs\b|\bs\b|\bm\b|\bl\b|\bxl\b', title_clean):
        return 'Size', None, 'Assigned to Size (regex match)'
    elif re.search(r'\bcolor\b|\bcolour\b|\bblack\b|\bwhite\b|\bblue\b', title_clean):
        return 'Color', None, 'Assigned to Color (regex match)'
    return primary_attr, secondary_attr, notes + '; Advanced QA applied'

# QA and review ambiguous titles
def qa_ambiguous_titles(pilot_file, full_file, output_file='qa_ambiguous_titles.xlsx', final_output='final_normalized_titles.xlsx', taxonomy_output='taxonomy_summary.xlsx'):
    logging.info("Milestone 3: Performing QA and Review...")
    
    pilot_df = pd.read_excel(pilot_file)
    full_df = pd.read_excel(full_file)
    combined_df = pd.concat([pilot_df, full_df], ignore_index=True)
    
    ambiguous_df = combined_df[combined_df['Notes'].str.contains('Ambiguous', na=False)].copy()
    
    for index, row in ambiguous_df.iterrows():
        title = str(row['Original_Title'])
        primary, secondary, new_notes = apply_qa_rules(title, row['Primary_Attribute'], row['Secondary_Attribute'], row['Notes'])
        ambiguous_df.at[index, 'Primary_Attribute'] = primary
        ambiguous_df.at[index, 'Secondary_Attribute'] = secondary
        ambiguous_df.at[index, 'Notes'] = new_notes
    
    ambiguous_df.to_excel(output_file, index=False)
    
    final_df = combined_df.copy()
    for index, row in ambiguous_df.iterrows():
        final_df.loc[final_df['Original_Title'] == row['Original_Title'], ['Primary_Attribute', 'Secondary_Attribute', 'Notes']] = row[['Primary_Attribute', 'Secondary_Attribute', 'Notes']]
    
    taxonomy_summary = final_df['Primary_Attribute'].value_counts().reset_index()
    taxonomy_summary.columns = ['Attribute', 'Count']
    taxonomy_summary.to_excel(taxonomy_output, index=False)
    
    final_df.to_excel(final_output, index=False)
    
    logging.info(f"QA complete. Outputs saved: {output_file}, {taxonomy_output}, {final_output}")
    return final_df, ambiguous_df

# Main execution
if __name__ == "__main__":
    pilot_file = 'pilot_batch.xlsx'
    full_file = 'full_batch.xlsx'
    qa_ambiguous_titles(pilot_file, full_file)