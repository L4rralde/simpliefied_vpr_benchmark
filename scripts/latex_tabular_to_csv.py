import re
import csv
import io
import argparse

def parse_latex_table_to_csv(latex_str):
    # Extract the content inside the tabular environment
    match = re.search(r'\\begin\{tabular\}(?:\{[^}]*\})?(.*?)\\end\{tabular\}', latex_str, re.DOTALL)
    if match:
        latex_str = match.group(1)
        
    # Remove comments
    latex_str = re.sub(r'%.*?(\n|$)', r'\n', latex_str)
    
    # Remove structural table lines
    rules = [r'\\hline', r'\\toprule', r'\\midrule', r'\\bottomrule', r'\\cline\{.*?\}']
    for rule in rules:
        latex_str = re.sub(rule, '', latex_str)
        
    # Split into rows using the LaTeX newline command
    raw_rows = [r.strip() for r in latex_str.split(r'\\')]
    clean_rows = [r for r in raw_rows if r]
    
    grid = {}
    header_depth = 1
    
    for r_idx, row_str in enumerate(clean_rows):
        # Temporarily protect escaped ampersands
        row_str = row_str.replace(r'\&', 'ESCAPED_AMP')
        cells = [c.strip() for c in row_str.split('&')]
        
        c_idx = 0
        for cell in cells:
            # Skip coordinates already occupied by a previous \multirow
            while grid.get((r_idx, c_idx)) is not None:
                c_idx += 1
                
            # Parse \multicolumn{cols}{align}{text}
            mc_match = re.search(r'\\multicolumn\{(\d+)\}\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}', cell)
            cols_span = int(mc_match.group(1)) if mc_match else 1
            cell_text = mc_match.group(2) if mc_match else cell
            
            # Parse \multirow{rows}{width}{text}
            mr_match = re.search(r'\\multirow\{(-?\d+)\}\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}', cell_text)
            rows_span = int(mr_match.group(1)) if mr_match else 1
            cell_text = mr_match.group(2) if mr_match else cell_text
            
            # Clean up residual LaTeX formatting
            cell_text = cell_text.replace('ESCAPED_AMP', '&').strip()
            
            # Update header depth heuristic if we are on the first row
            if r_idx == 0:
                if rows_span > 1:
                    header_depth = max(header_depth, rows_span)
                if cols_span > 1 and header_depth == 1:
                    header_depth = 2

            # Populate grid: store a tuple of (text, is_top_left_of_merge)
            for i in range(cols_span):
                for j in range(rows_span):
                    is_master = (i == 0 and j == 0)
                    grid[(r_idx + j, c_idx + i)] = (cell_text, is_master)
                    
            c_idx += cols_span

    if not grid:
        return ""
        
    max_r = max(k[0] for k in grid.keys())
    max_c = max(k[1] for k in grid.keys())
    
    csv_data = []
    
    # 1. Process and flatten the Header rows
    header_row = []
    for c in range(max_c + 1):
        compound_header = []
        for r in range(header_depth):
            # Extract just the text from the tuple
            val = grid.get((r, c), ("", False))[0]
            if val and (not compound_header or compound_header[-1] != val):
                compound_header.append(val)
        header_row.append("-".join(compound_header))
    
    if header_row:
        csv_data.append(header_row)
        
    # 2. Process the remaining Data rows
    for r in range(header_depth, max_r + 1):
        row_data = []
        for c in range(max_c + 1):
            val, is_master = grid.get((r, c), ("", False))
            # Only print the text if it's an independent cell or the master of a merged cell
            if is_master:
                row_data.append(val)
            else:
                row_data.append("")
                
        csv_data.append(row_data)
        
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(csv_data)
    
    return output.getvalue()


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a LaTeX tabular environment to a CSV file.")
    parser.add_argument("input_file", help="Path to the input .tex file")
    parser.add_argument("-o", "--output", help="Path to the output .csv file (defaults to stdout if not provided)", default=None)
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            latex_content = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{args.input_file}' was not found.", file=sys.stderr)
        return

    csv_output = parse_latex_table_to_csv(latex_content)

    if not csv_output.strip():
        print("Warning: No tabular data was found or parsed from the input file.", file=sys.stderr)

    if args.output:
        with open(args.output, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_output)
        print(f"Successfully wrote CSV data to '{args.output}'")
    else:
        # Print to console if no output file is specified
        print(csv_output)


if __name__ == "__main__":
    main()