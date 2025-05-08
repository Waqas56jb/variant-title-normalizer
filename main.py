import os
import logging
from milestone1_pilot_batch import process_pilot_batch
from milestone2_full_batch import process_full_batch
from milestone3_qa_review import review_and_qa

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Input file
    variant_data_file = "variant_data.xlsx"
    output_dir = "output"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Milestone 1: Pilot Batch (4,000 titles as requested)
    logging.info("Starting Milestone 1: Pilot Batch")
    pilot_output = os.path.join(output_dir, "milestone1_pilot_output.xlsx")
    process_pilot_batch(variant_data_file, pilot_output, sample_size=4000)
    
    # Milestone 2: Full Batch (remaining 11,500+ titles)
    logging.info("Starting Milestone 2: Full Batch")
    full_output = os.path.join(output_dir, "milestone2_full_output.xlsx")
    process_full_batch(variant_data_file, pilot_output, full_output)
    
    # Milestone 3: Review & QA
    logging.info("Starting Milestone 3: Review & QA")
    final_output = os.path.join(output_dir, "milestone3_final_output.xlsx")
    review_and_qa(full_output, final_output)
    
    logging.info("All milestones completed successfully.")

if __name__ == "__main__":
    main()