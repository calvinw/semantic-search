#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

def truncate_csv(input_file: str, output_file: str, max_lines: int = 500) -> None:
    """
    Truncate a CSV file to a specified number of lines while preserving the header.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the truncated CSV file
        max_lines: Maximum number of lines to keep (including header)
    """
    try:
        with open(input_file, 'r', newline='') as infile:
            # Read the header first
            reader = csv.reader(infile)
            header = next(reader)
            
            # Read up to max_lines-1 data rows (accounting for header)
            data_rows = []
            for i, row in enumerate(reader):
                if i >= max_lines - 1:
                    break
                data_rows.append(row)
        
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            writer.writerows(data_rows)
            
        print(f"Successfully truncated {input_file} to {max_lines} lines")
        print(f"Output saved to {output_file}")
            
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        exit(1)
    except PermissionError:
        print(f"Error: Permission denied when accessing '{input_file}' or '{output_file}'")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Truncate a CSV file to a specified number of lines.")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("output_file", help="Path to save the truncated CSV file")
    parser.add_argument("-n", "--lines", type=int, default=500,
                      help="Maximum number of lines in output (including header). Default: 500")
    
    args = parser.parse_args()
    truncate_csv(args.input_file, args.output_file, args.lines)
