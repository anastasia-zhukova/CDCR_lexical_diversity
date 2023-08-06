## Explain how this data works

The original data is organized by splits. Each split has a json file a csv file.

- The json file (features) contain the following variables: index, mid, midid, incidentid, headline, text, date,split
- The csv file contain that following variables: headline a, headline b and coref.

- MID refers to a conflict that lasts for many years.
- Incidentid refers to a time-specific event in the context of one MID, but it is a different event in time.

However,an example like this is confusing:

midid: 3561
3561003	Russian official warn of potential fight between Russia Georgia	Russian official warn of potential fight between Russia Georgia	1992-10-04
3561003	Moscow and Tblisi exchange threats	Moscow and Tblisi exchange threats	1992-10-03
3561004	Caucasus near war	Caucasus near war	1992-10-05
3561004	Russians detain helicopters carrying wounded soldiers	Russians detain helicopters carrying wounded soldiers	1992-10-05
The midid is the same, but the incidentid does not make a big distinction between both events.

The first example of the incidentid 3561004 could be related to either incident within that mid.

-----

After checking, all the midid's for pairs labeled as coref coincide

Keypair 201MID252MID belong to the same midid but are labeled as coref=0, which is weird.

Ultimately there is not a big distinction between midid and incidentid to build the paraphrase pairs.


In order to get the final data, it is necessary to load it using the huggingface datasets library from its path like this:

dataset_test = DatasetDict.load_from_disk('HoW_DatasetDict')