from datasets import load_dataset

train_data = load_dataset('codeparrot/codeparrot-clean-valid', split='train')
train_data.to_json("codeparrot_data.json", lines=True)