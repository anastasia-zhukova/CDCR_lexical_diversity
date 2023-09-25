import pandas as pd
from datasets import load_dataset,Dataset, DatasetDict

df = pd.read_feather('HoWMerge.fth')
df_clean = df.reset_index(drop=True).reset_index()
df_clean.rename(columns={"texta":"sentence1",
                         "textb":"sentence2",
                         "coref":"label",
                         "index":"idx"},inplace=True)


df_train = df_clean.query("split=='train'").drop(["split","level_0"],axis=1)
df_test = df_clean.query("split=='test'").drop(["split","level_0"],axis=1)
df_validation = df_clean.query("split=='valid'").drop(["split","level_0"],axis=1)

trainds = Dataset.from_pandas(df_train,preserve_index=False)
testds = Dataset.from_pandas(df_test,preserve_index=False)
valds = Dataset.from_pandas(df_validation,preserve_index=False)

ds = DatasetDict()

ds['train'] = trainds
ds['validation'] = valds
ds['test'] = testds

# Save the DatasetDict to disk
ds.save_to_disk("HoW_DatasetDict")