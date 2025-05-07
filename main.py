import logging
from milestone1_pilot_batch import process_pilot_batch
from milestone2_full_batch import process_full_batch
from milestone3_qa_review import qa_ambiguous_titles

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Main execution
if __name__ == "__main__":
    input_file = 'variant_data.xlsx'
    
    logging.info("Starting Milestone 1: Pilot Batch")
    pilot_df = process_pilot_batch(input_file, pilot_size=1500, output_file='pilot_batch.xlsx')
    
    logging.info("Starting Milestone 2: Full Batch")
    full_df = process_full_batch(input_file, pilot_size=1500, output_file='full_batch.xlsx')
    
    logging.info("Starting Milestone 3: QA & Review")
    final_df, ambiguous_df = qa_ambiguous_titles('pilot_batch.xlsx', 'full_batch.xlsx')
    
    logging.info("All milestones completed successfully.")