"""
Example script showing how to load filtered topics from a YAML config file
and use them with the CDCR_Dataset class.
"""

import yaml
from CL_COREF.dataset import CDCR_Dataset, CL_Dataset

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Example 1: Using CDCR_Dataset directly with filtered topics
def example_cdcr_dataset_with_filter():
    # Load config
    config = load_config('config_example.yaml')
    filtered_topics = config.get('filtered_topics', [])

    # Create dataset with topic filtering
    dataset = CDCR_Dataset(
        name="example_dataset",
        data_path="CL_COREF/data/ECBplus/train",
        context_method="full_context",
        special_mention_tokens=("[E]", "[/E]"),
        predicted_topics=False,
        filtered_topics=filtered_topics  # Pass the filtered topics from config
    )

    print(f"Loaded {len(dataset.mentions)} mentions")
    print(f"Topics in dataset: {list(dataset.mentions_by_topic.keys())}")

# Example 2: Using CL_Dataset with filtered topics
def example_cl_dataset_with_filter(tokenizer, sentence_transformer_model):
    # Load config
    config = load_config('config_example.yaml')
    filtered_topics = config.get('filtered_topics', [])

    # Create dataset with topic filtering
    dataset = CL_Dataset(
        name="example_dataset",
        data_paths="CL_COREF/data/ECBplus/train",
        context_method="full_context",
        tokenizer=tokenizer,
        special_mention_tokens=("[E]", "[/E]"),
        mode="train",
        additional_sampling=False,
        predicted_topics=False,
        sentence_transformer_model=sentence_transformer_model,
        filtered_topics=filtered_topics  # Pass the filtered topics from config
    )

    print(f"Loaded {len(dataset.cdcr_dataset.mentions)} mentions")

# Example 3: Load all topics (empty or None filtered_topics)
def example_load_all_topics():
    # When filtered_topics is empty or None, all topics are loaded
    dataset = CDCR_Dataset(
        name="example_dataset",
        data_path="CL_COREF/data/ECBplus/train",
        context_method="full_context",
        filtered_topics=[]  # Empty list = load all topics
    )

    # Or simply omit the parameter (defaults to None, which becomes [])
    dataset2 = CDCR_Dataset(
        name="example_dataset",
        data_path="CL_COREF/data/ECBplus/train",
        context_method="full_context"
    )

    print(f"All topics loaded: {len(dataset.mentions)} mentions")

if __name__ == "__main__":
    # Run examples (you'll need to provide actual tokenizer and model)
    example_cdcr_dataset_with_filter()
