import logging
import os
from milestone2 import process_full_batch
from milestone3 import process_qa_review

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main workflow for Variant Title Normalization."""
    logging.info("Starting Variant Title Normalization Workflow")
    
    # File paths (relative to script directory)
    full_input = "variant_data.xlsx"
    pilot_output = "pilot_normalized_titles.xlsx"
    full_output = "full_normalized_titles.xlsx"
    final_output = "normalized_variants.xlsx"
    taxonomy = "taxonomy.json"
    summary = "qa_summary.json"
    
    # Milestone 2: Full Batch
    logging.info("Executing Milestone 2")
    process_full_batch(full_input, full_output, taxonomy)
    
    # Milestone 3: Review & QA
    logging.info("Executing Milestone 3")
    process_qa_review(pilot_output, full_output, final_output, taxonomy, summary)
    
    logging.info("Workflow completed successfully")

if __name__ == "__main__":
    main()