import pandas as pd
#Load data
df_features = pd.read_feather('df_features.fth')
df_labels = pd.read_feather('df_labels.fth')
# Rename columns
df_text_a = df_features[['index','headline']].rename(columns={'index':'headlinea','headline':'texta'})
df_text_b = df_features[['index','headline']].rename(columns={'index':'headlineb','headline':'textb'})

#Create key
df_merged = df_labels.merge(df_text_a, on=['headlinea'],how='left').merge(df_text_b, on=['headlineb'],how='left')
df_merged['pairkey'] = df_merged.apply(lambda row: min(row['headlinea'],row['headlineb'])+max(row['headlinea'],row['headlineb']),axis=1)

#Drop repeated pairs of events
df_merged_nodups= df_merged.drop_duplicates(subset=['pairkey'])

# Remove identical texts
df_merged_nodups_final= df_merged_nodups.query('texta!=textb').copy()

#Turn coref label into integer
df_merged_nodups_final["coref"]= df_merged_nodups_final["coref"].astype(int)

#Leave relevant columns only
df_merged_nodups_final = df_merged_nodups_final.drop(["index","headlinea","headlineb","pairkey"],axis=1).reset_index()

df_merged_nodups_final.to_feather("HoWMerge.fth")