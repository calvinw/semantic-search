import csv
import io

def filter_dresses(input_file, output_file):
    """
    Read through a CSV file of dress data and remove all Baby/Children dresses.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Store all rows and headers
        headers = reader.fieldnames
        rows = [row for row in reader if row['index_group_name'] != 'Baby/Children']
    
    # Write filtered data to output file
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

# Example usage
if __name__ == "__main__":
    input_file = "hm-dresses.csv"  # Replace with your input file name
    output_file = "filtered-hm-dresses.csv"  # Replace with desired output file name
    filter_dresses(input_file, output_file)
