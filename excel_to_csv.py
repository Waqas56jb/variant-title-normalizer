import pandas as pd
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to convert Excel to CSV
def convert_excel_to_csv(input_files, output_prefix='converted_'):
    """
    Convert a list of Excel files to CSV with modified filenames.
    
    Args:
        input_files (list): List of Excel file paths to convert.
        output_prefix (str): Prefix to add to output CSV filenames (default: 'converted_').
    """
    for excel_file in input_files:
        try:
            # Check if the file exists
            if not os.path.exists(excel_file):
                logging.error(f"File '{excel_file}' not found. Skipping conversion.")
                continue
            
            # Generate output CSV filename
            base_name = os.path.splitext(excel_file)[0]  # Remove .xlsx
            csv_file = f"{output_prefix}{base_name}.csv"  # Add prefix and .csv
            
            # Read Excel file
            logging.info(f"Reading Excel file: {excel_file}")
            df = pd.read_excel(excel_file)
            
            # Save as CSV
            logging.info(f"Converting to CSV: {csv_file}")
            df.to_csv(csv_file, index=False)
            
            logging.info(f"Successfully converted '{excel_file}' to '{csv_file}'")
            
        except Exception as e:
            logging.error(f"Failed to convert '{excel_file}': {e}")
            continue

# Main execution
if __name__ == "__main__":
    # List of Excel files to convert
    excel_files = [
        'pilot_batch.xlsx',
        'full_batch.xlsx',
        'qa_ambiguous_titles.xlsx',
        'taxonomy_summary.xlsx',
        'final_normalized_titles.xlsx'
    ]
    
    # Convert all Excel files to CSV
    convert_excel_to_csv(excel_files, output_prefix='converted_')