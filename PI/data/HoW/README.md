# Headlines of War (HoW) Dataset

## Overview

[**Headlines of War (HoW)**](https://aclanthology.org/2020.aespen-1.7/) is a dataset comprised of a diverse collection of news story headlines. These headlines serve as paraphrases pairs with biased and lexically diverse language. Below, you'll find an overview of this dataset and instructions on how to access and process it.

### Dataset Details

- **Source**: [Download HoW Dataset](https://doi.org/10.7910/DVN/8TEG5R)
- **Data Structure**: Splits containing JSON and CSV files.

#### JSON File (Features)

The JSON file, referred to as "features," includes the following variables:

- `index`: A unique identifier for each entry.
- `mid`: Denotes a conflict that spanned multiple years.
- `midid`: A specific identifier for each mid.
- `incidentid`: Identifies a time-specific event within a mid, representing different events occurring at distinct time points.
- `headline`: The headline of the news story.
- `text`: The full text of the news story.
- `date`: The date associated with the news story.
- `split`: The categorization of data into specific sets.

#### CSV File

The CSV file contains the following variables:

- `headline a`: One of the headlines.
- `headline b`: Another headline.
- `coref`: Information regarding coreference between the two headlines.

Key Clarifications:

- **MID (Mid)**: Refers to a conflict that extended over several years.
- **Incidentid (Incident ID)**: Identifies a specific event within a MID, representing different events occurring at various time points.

## Data Processing

To make effective use of this dataset and export it as a Hugging Face dataset, follow these steps:

1. **Load Data**: Begin by loading the source data using the `0_load_data.py` script.
2. **Unify Data**: Use the `1_unify_data.py` script to unify and structure the dataset.
3. **Prepare Data for Hugging Face**: Finally, prepare the data for Hugging Face by running the `2_prepare_data_huggingface.py` script.

After successfully processing the data, you can access it using the Hugging Face library with the following command:

```python
from datasets import load_dataset

# Load the HoW dataset
dataset_test = DatasetDict.load_from_disk('HoW_DatasetDict')
