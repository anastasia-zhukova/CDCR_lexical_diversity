import json
import pandas as pd
import re

import requests
import zipfile
import io
# Extract the contents of the zip file to a temporary directory
import tempfile
import os
import shutil

# Download the zip file from the URL
temp_dir = tempfile.mkdtemp()

def download_unzip_files(temp_dir):
    url = 'https://owncloud.gwdg.de/index.php/s/wqUJmN53BYZK9ff/download'
    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall(temp_dir)

def delete_temporary_files(temp_dir):
    shutil.rmtree(temp_dir)

# Opening JSON file

def remove_blanks_concat(txt_list):
    clean_list = [x for x in txt_list if x!="_BLANK"]
    concat = " ".join(clean_list)
    
    return concat

def remove_double_spaces(txt):
    clean_txt = re.sub(' +', ' ', txt).strip()
    return clean_txt


# Process json files
download_unzip_files(temp_dir)


dict_data={}
list_df_labels=[]

for root, dirs, files in os.walk(temp_dir):
    for filename in files:
        # Do something with each file
        file_path = os.path.join(root, filename)
        category= filename.split("_")[1]
        if "json" in filename:
            f = open(file_path)
            data = json.load(f)
            dict_data[category] = data

        if "csv" in filename:
            df_label = pd.read_csv(file_path)
            df_label["split"] = category
            list_df_labels.append(df_label)


# turn headlines into table
list_dfs= []
for k,v in dict_data.items():
    df= pd.DataFrame.from_dict(v , orient='index')
    df["split"] = k
    list_dfs.append(df)


df = pd.concat(list_dfs)
df_labels = pd.concat(list_df_labels)

df["text"]= df["text"].apply(remove_blanks_concat)
df["text"]= df["text"].apply(remove_double_spaces)
df["headline"]= df["headline"].apply(remove_double_spaces)

# delete_temporary_files(temp_dir)
    
# Export files
df.reset_index().to_feather("df_features.fth")
df_labels.reset_index().to_feather('df_labels.fth')