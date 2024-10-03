import re
import pandas as pd

# Define the year
YEAR = 2022

# Utility functions for string processing
def quotemeta(string):
    return re.sub(r'(\W)', r'\\\1', string)

def remove_punct(string):
    return re.sub(r'[^\w\s]', ' ', string)

def collapse_whitespace(string):
    return re.sub(r'\s+', ' ', string)

# Load data
crs = pd.read_csv(f"crs_{YEAR}_predictions.csv")
original_names = crs.columns.tolist()[:95]

#  Remove CERF transactions that have the CERF as a channel rather than as donor
crs = crs[~((crs['channel_name'] == "Central Emergency Response Fund") & 
            (crs['donor_name'] != "Central Emergency Response Fund"))]

# Follow OECD’s description of humanitarian aid
crs['humanitarian'] = crs['purpose_code'].isin([720, 72010, 72040, 72050, 730, 73010, 74020])

# Set blanks to false and 0
blanks = ["", "-"]
blank_indices = crs[(crs['project_title'].isin(blanks)) &
                    (crs['short_description'].isin(blanks)) &
                    (crs['long_description'].isin(blanks))].index
cols_to_update = [
    'Crisis finance confidence', 'Crisis finance predicted', 'PAF confidence',
    'PAF predicted', 'AA confidence', 'AA predicted', 'Direct confidence',
    'Direct predicted', 'Indirect confidence', 'Indirect predicted', 
    'Part confidence', 'Part predicted'
]

for col in cols_to_update:
    crs.loc[blank_indices, col] = 0 if 'confidence' in col else False

# Set names
col_rename_map = {
    "Crisis finance predicted": "Crisis finance predicted ML",
    "Crisis finance confidence": "Crisis finance confidence ML",
    "PAF predicted": "PAF predicted ML",
    "PAF confidence": "PAF confidence ML",
    "AA predicted": "AA predicted ML",
    "AA confidence": "AA confidence ML",
    "Direct predicted": "Direct predicted ML",
    "Direct confidence": "Direct confidence ML",
    "Indirect predicted": "Indirect predicted ML",
    "Indirect confidence": "Indirect confidence ML",
    "Part predicted": "Part predicted ML",
    "Part confidence": "Part confidence ML"
}
crs.rename(columns=col_rename_map, inplace=True)

# Construct total crisis financing
crs['Crisis finance identified'] = False

# Include financing under channel codes
crs.loc[crs['channel_code'].isin([
    21016, 21018, 21029, 23501, 41121, 41127, 41130, 41147, 41315, 41321,
    41403, 43003, 43005, 43006, 47123, 47137, 47502
]), 'Crisis finance identified'] = True

# Include financing under purpose codes
crs.loc[crs['purpose_code'].isin([
    12264, 15220, 15240, 15250, 15261, 43060
]), 'Crisis finance identified'] = True

# Set boolean fields
crs['Crisis finance eligible'] = crs['purpose_code'].isin([
    111, 11110, 11120, 11130, 11182, 112, 11220, 11230, 11231, 11232,
    11240, 11250, 11260, 121, 12110, 12191, 122, 12220, 12230, 12240,
    12250, 12261, 12262, 12263, 12281, 130, 13010, 13020, 13030, 13081,
    140, 14010, 14015, 14020, 14021, 14022, 14030, 14031, 14032, 14040,
    14050, 14081, 151, 15110, 15111, 15114, 15142, 15160, 15170, 15190,
    152, 160, 16010, 16020, 16050, 16062, 210, 21010, 21020, 21030, 21040,
    21050, 21061, 21081, 240, 24010, 24020, 24030, 24040, 24050, 24081,
    311, 31110, 31120, 31130, 31140, 31191, 321, 32130, 410, 41010, 430,
    43010, 43030, 43040, 43071, 43072, 43082, 510, 51010, 520, 52010, 600,
    60010, 60020, 60030, 60040, 60061, 60062, 60063
])

# Combine textual columns and preprocess text
textual_cols_for_classification = ['project_title', 'short_description', 'long_description']
crs['text'] = crs[textual_cols_for_classification].apply(lambda row: ' '.join(row.dropna().values.astype(str)), axis=1)
crs['text'] = crs['text'].str.lower().str.strip().apply(remove_punct).apply(collapse_whitespace)

# Exclude transactions containing bad relief keywords
bad_relief_keywords = [
    "Comic Relief", "Sport Relief", "Medical Relief Society", 
    "Assemblies of God Relief and Development Services", "Catholic Relief Services", 
    "AIDS Relief", "World Bicycle Relief", "KSrelief", "The RELIEF Centre", 
    "Relief International"
]
bad_relief_regex = r'\b(' + '|'.join(map(re.escape, bad_relief_keywords)) + r')\b'

crs['Contains bad relief'] = crs['text'].str.contains(bad_relief_regex, case=False, regex=True)
crs['Contains debt relief'] = crs['text'].str.contains(r'\bdebt relief\b', case=False, regex=True)
crs['Contains relief'] = crs['text'].str.contains(r'\brelief\b', case=False, regex=True)

# Keywords from external CSV
keywords = pd.read_csv("data/keywords.csv")
keywords['keyword'] = keywords['keyword'].apply(lambda x: quotemeta(remove_punct(x.lower().strip())))
keywords_cf = keywords[keywords['category'] == "CF"]['keyword'].tolist()
keywords_paf = keywords[keywords['category'] != "CF"]['keyword'].tolist()
keywords_aa = keywords[~keywords['category'].isin(["CF", "PAF"])]["keyword"].tolist()

cf_regex = r'\b(' + '|'.join(keywords_cf) + r')\b'
paf_regex = r'\b(' + '|'.join(keywords_paf) + r')\b'
aa_regex = r'\b(' + '|'.join(keywords_aa) + r')\b'

# Apply keyword matching
crs['Crisis finance keyword match'] = crs['text'].str.contains(cf_regex, case=False, regex=True)
crs['PAF keyword match'] = crs['text'].str.contains(paf_regex, case=False, regex=True)
crs['AA keyword match'] = crs['text'].str.contains(aa_regex, case=False, regex=True)

# Additional logic for relief
crs.loc[(crs['Crisis finance keyword match'] == False) & (crs['Contains relief'] == True) &
        (crs['Contains bad relief'] == False) & (crs['Contains debt relief'] == False), 
        'Crisis finance keyword match'] = True

# Capture activities matching only debt relief
crs['Only debt relief'] = (crs['Contains debt relief'] == True) & (crs['Crisis finance keyword match'] == False)

# Crisis finance determination logic
crs['Crisis finance determination'] = "No"
crs.loc[(crs['Crisis finance identified'] | crs['Crisis finance eligible']) & 
        (crs['Crisis finance keyword match']), 'Crisis finance determination'] = \
    crs.apply(lambda x: "Yes" if x['Crisis finance predicted ML'] else "Review", axis=1)

# Adjust for relief and debt cases
crs.loc[crs['Crisis finance identified'], 'Crisis finance determination'] = "Yes"
crs.loc[(crs['Crisis finance determination'] == "Yes") & crs['Only debt relief'], 'Crisis finance determination'] = "Review"

# PAF and AA determinations
crs['PAF determination'] = crs.apply(lambda x: "Yes" if (x['PAF keyword match'] and x['PAF predicted ML']) 
                                     else ("Review" if x['PAF keyword match'] else "No"), axis=1)
crs['AA determination'] = crs.apply(lambda x: "Yes" if (x['AA keyword match'] and x['AA predicted ML']) 
                                     else ("Review" if x['AA keyword match'] else "No"), axis=1)

# Interrelationships
crs.loc[(crs['PAF determination'] != "Yes") & (crs['AA determination'] == "Yes"), 'PAF determination'] = "Yes"
crs.loc[(crs['PAF determination'] != "Yes") & (crs['AA determination'] == "Review"), 'PAF determination'] = "Review"

crs.loc[(crs['Crisis finance determination'] != "Yes") & (crs['PAF determination'] == "Yes"), 'Crisis finance determination'] = "Yes"
crs.loc[(crs['Crisis finance determination'] != "Yes") & (crs['PAF determination'] == "Review"), 'Crisis finance determination'] = "Review"

# Set 'Crisis finance determination' to "Yes" for humanitarian cases
crs.loc[crs['humanitarian'], 'Crisis finance determination'] = "Yes"

# Set 'humanitarian' to True if 'AA determination' is "Yes" and 'humanitarian' is currently False
crs.loc[(~crs['humanitarian']) & (crs['AA determination'] == "Yes"), 'humanitarian'] = True

# Descriptive statistics for the three determination columns (similar to R's describe)
print(crs['Crisis finance determination'].value_counts())
print(crs['PAF determination'].value_counts())
print(crs['AA determination'].value_counts())

# Handling predicted columns
predicted_cols = [col for col in crs.columns if col.endswith("predicted ML")]
for col in predicted_cols:
    crs.loc[crs[col] == False, col] = ""

# Handling search columns (those ending with "match")
search_cols = [col for col in crs.columns if col.endswith("match")]
for col in search_cols:
    crs.loc[crs[col] == False, col] = ""

# Handling other boolean columns
other_bool_cols = ['humanitarian', 'Crisis finance identified', 'Crisis finance eligible', 'Contains debt relief']
for col in other_bool_cols:
    crs.loc[crs[col] == False, col] = ""

# Select specific columns to keep in the final output
keep_columns = original_names + [
    'humanitarian',
    'Crisis finance identified',
    'Crisis finance eligible',
    'Crisis finance determination',
    'Crisis finance keyword match',
    'Crisis finance predicted ML',
    'Crisis finance confidence ML',
    'PAF determination',
    'PAF keyword match',
    'PAF predicted ML',
    'PAF confidence ML',
    'AA determination',
    'AA keyword match',
    'AA predicted ML',
    'AA confidence ML',
    'Direct predicted ML',
    'Direct confidence ML',
    'Indirect predicted ML',
    'Indirect confidence ML',
    'Part predicted ML',
    'Part confidence ML'
]

# Keep only certain columns
crs = crs[keep_columns]

# Save the final result to CSV
output_file = f"crs_{YEAR}_cdp_automated_py.csv"
crs.to_csv(output_file, index=False)
