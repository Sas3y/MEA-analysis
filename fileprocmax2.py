# Makes files smaller and easier to process

def format_float_custom(value):
    """
    no trailing zeros, no scientific notation.
    Leans on fast, C-implemented Python built-ins.
    """
    rounded_value = round(value, 3)
    
    s = f"{rounded_value:.{3}f}"

    s = s.rstrip('0') 

    if s.endswith('.'): 
        s = s[:-1] 
    return s


def process_large_file_python_minimal(input_filename, output_filename, header_lines_to_skip=2, data_lines_to_process=None):
    """
    Processes a large text file, skipping headers, normalizing whitespace,
    and writing a specified number of data lines. Timestamps are custom formatted.
    """
    lines_written = 0

    try:
        with open(input_filename, 'r', encoding='latin-1') as infile, \
             open(output_filename, 'w', encoding='utf-8') as outfile: #conversion to utf-8 seems smart
            
            # Skip header lines
            for _ in range(header_lines_to_skip):
                header_line = infile.readline()
                print(header_line)
                if not header_line:
                    # This warning is important, so it's kept.
                    print(f"Warning: Input file is shorter than the specified {header_lines_to_skip} initial lines to skip.")
                    return # Stop processing if headers are incomplete as expected

            # Process data lines
            for line in infile: # Changed from enumerate, as line_num is not used
                stripped_line = line.strip()

                # Filter out empty lines or lines not starting with a digit or '-' followed by digit
                #if not stripped_line or not (stripped_line[0].isdigit() or \
                #                             (stripped_line.startswith('-') and len(stripped_line) > 1 and stripped_line[1].isdigit())):
                #    continue

                #if data_lines_to_process is not None and lines_written >= data_lines_to_process:
                    # Informational print about reaching limit removed for minimality
                #    break
                
                split_values = stripped_line.split()
                normalized_values = []
                for col_idx, item_str in enumerate(split_values):
                    try:
                        value = float(item_str)
                        if col_idx == 0: # First column (timestamp)
                            formatted_value = format_float_custom(value)
                        else: # Other columns
                            formatted_value = f"{value:g}" # :g is generally efficient
                        normalized_values.append(formatted_value)
                    except ValueError:
                        normalized_values.append(item_str) # Keep non-float items as is

                outfile.write('\t'.join(normalized_values) + '\n')
                lines_written += 1
                
                # Progress printing removed for performance
                if lines_written % 1000000 == 0: #every 33secs
                    print(f"Processed {lines_written}")

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback # Import traceback only when an exception occurs
        traceback.print_exc()

if __name__ == "__main__":
    #IMPORTANT HERE CHANGE FILE INPUT FILE CHANGE HERE IMPORTANT HERE CHANGE FILE INPUT FILE CHANGE HERE IMPORTANT
    input_file = "Dana100uM0001-a.txt" # Replace with your input file
    output_file = "325Dana100-1p.tsv"  # Replace with your desired output file

    process_large_file_python_minimal(input_file, output_file,
                                      header_lines_to_skip=4
                                      #,data_lines_to_process=100000
                                      ) # Example: process 10,000 data lines
    
    print(f"Script finished.")