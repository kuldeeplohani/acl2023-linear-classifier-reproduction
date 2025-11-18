from datasets import load_dataset
import os

# List of LexGLUE datasets used in your repo
datasets = ['ecthr_a', 'ecthr_b', 'scotus', 'eurlex', 'ledgar', 'unfair_tos']

# Choose your data format ('linear', 'nn', or 'hier')
data_format = 'linear'
data_dir_prefix = f"data_{data_format}"

for data in datasets:
    print(f"Processing: {data}")

    # Load the dataset
    dataset = load_dataset("coastalcph/lex_glue", data, split="train", trust_remote_code=True)

    # Detect label field
    label_names = None
    if "labels" in dataset.features:
        try:
            label_names = dataset.features["labels"].feature.names
        except Exception:
            # Some datasets have nested lists of ints, no .feature.names
            label_names = [str(i) for i in range(max(max(dataset["labels"])) + 1)]
    elif "label" in dataset.features:
        label_names = dataset.features["label"].names
    else:
        raise ValueError(f"No label field found for {data}")

    # Output directory (e.g., data_linear/unfair_tos)
    output_dir = os.path.join(data_dir_prefix, data)
    os.makedirs(output_dir, exist_ok=True)

    # Write classes.txt
    classes_path = os.path.join(output_dir, "classes.txt")
    with open(classes_path, "w") as f:
        for name in label_names:
            f.write(name + "\n")

    print(f"Created {classes_path} ({len(label_names)} labels)\n")
