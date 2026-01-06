import json
from datasets import load_dataset
import sys

# Output file name
output_file = "codeparrot_data_cleaned.json"

# Load the dataset
print("Loading dataset...")
try:
    # Try loading from the user's path if it was a file, but here we likely want to load from hub or the local json file the user mentioned
    # The user mentioned /pfs/ziqijin/Megatron-LM/examples/qwen3/06btest/codeparrot_data.json
    # We will read that file line by line, clean it, and write it back.
    
    input_file = "codeparrot_data.json"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    print(f"Processing {input_file} -> {output_file}")
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            # transform to just text
            if 'content' in data:
                new_data = {'text': data['content']}
                f_out.write(json.dumps(new_data) + "\n")
            elif 'text' in data:
                new_data = {'text': data['text']}
                f_out.write(json.dumps(new_data) + "\n")
                
    print("Done!")

except Exception as e:
    print(f"Error processing data: {e}")