list.of.packages <- c("data.table", "ggplot2", "Hmisc", "tidyverse", "stringr", "rstudioapi")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

YEAR = 2022

setwd(dirname(getActiveDocumentContext()$path))
setwd("../")

quotemeta <- function(string) {
  str_replace_all(string, "(\\W)", "\\\\\\1")
}

remove_punct = function(string){
  str_replace_all(string, "[[:punct:]]", " ")
}

collapse_whitespace = function(string){
  str_replace_all(string, "\\s+", " ")
}

crs = fread(paste0("crs_",YEAR,"_predictions.csv"))
original_names = names(crs)[1:95]

#  Remove CERF transactions that have the CERF as a channel rather than as donor. These
# are likely to be reporting errors and might cause double counting.
crs = subset(crs, 
             channel_name!="Central Emergency Response Fund" | donor_name=="Central Emergency Response Fund"
             )

# Follow the OECD’s description of humanitarian aid 
crs$humanitarian = 
  crs$purpose_code %in% c(720, 72010, 72040, 72050, 730, 73010, 74020)

# Set blanks to false and 0
blanks = c("", "-")
blank_indices = which(crs$project_title %in% blanks & crs$short_description %in% blanks & crs$long_description %in% blanks)
crs$`Crisis finance confidence`[blank_indices] = 0
crs$`Crisis finance predicted`[blank_indices] = F
crs$`PAF confidence`[blank_indices] = 0
crs$`PAF predicted`[blank_indices] = F
crs$`AA confidence`[blank_indices] = 0
crs$`AA predicted`[blank_indices] = F
crs$`Direct confidence`[blank_indices] = 0
crs$`Direct predicted`[blank_indices] = F
crs$`Indirect confidence`[blank_indices] = 0
crs$`Indirect predicted`[blank_indices] = F
crs$`Part confidence`[blank_indices] = 0
crs$`Part predicted`[blank_indices] = F


# Set PAF confidence equal to AA confidence if AA predicted and PAF not
# crs$`PAF confidence`[which(crs$`AA predicted` & !crs$`PAF predicted`)] =
#   crs$`AA confidence`[which(crs$`AA predicted` & !crs$`PAF predicted`)]
# crs$`PAF predicted`[which(crs$`AA predicted`)] = T

# Set CF confidence equal to PAF confidence if PAF predicted and CF not
# crs$`Crisis finance confidence`[which(crs$`PAF predicted` & !crs$`Crisis finance predicted`)] =
#   crs$`PAF confidence`[which(crs$`PAF predicted` & !crs$`Crisis finance predicted`)]
# crs$`Crisis finance predicted`[which(crs$`PAF predicted`)] = T

setnames(crs,
         c(
           "Crisis finance predicted",
           "Crisis finance confidence",
           "PAF predicted",
           "PAF confidence",
           "AA predicted",
           "AA confidence",
           "Direct predicted",
           "Direct confidence",
           "Indirect predicted",
           "Indirect confidence",
           "Part predicted",
           "Part confidence"
           ),
         c(
           "Crisis finance predicted ML",
           "Crisis finance confidence ML",
           "PAF predicted ML",
           "PAF confidence ML",
           "AA predicted ML",
           "AA confidence ML",
           "Direct predicted ML",
           "Direct confidence ML",
           "Indirect predicted ML",
           "Indirect confidence ML",
           "Part predicted ML",
           "Part confidence ML"
         )
         )

# Construct total crisis financing
crs$`Crisis finance identified` = F

## Include all the financing under channel codes
crs$`Crisis finance identified`[
  which(
    crs$channel_code %in% c(
      21016 # International Committee of the Red Cross - 
      ,21018 # International Federation of Red Cross and Red Crescent Societies - 
      ,21029 # Doctors Without Borders - 
      ,23501 # National Red Cross and Red Crescent Societies -
      ,41121 # United Nations Office of the United Nations High Commissioner for Refugees - 
      ,41127 # United Nations Office of Co-ordination of Humanitarian Affairs - 
      ,41130 # United Nations Relief and Works Agency for Palestine Refugees in the Near East - 
      ,41147 # Central Emergency Response Fund - 
      ,41315 # United Nations Office for Disaster Risk Reduction - 
      ,41321 # World Health Organisation – Strategic Preparedness and Response Plan -
      ,41403 # COVID-19 Response and Recovery Multi-Partner Trust Fund - 
      ,43003 # International Monetary Fund – Subsidization of Emergency Post Conflict Assistance/Emergency Assistance for Natural Disasters for PRGTeligible members - 
      ,43005 # International Monetary Fund – Post-Catastrophe Debt Relief Trust - 
      ,43006 # Catastrophe Containment and Relief Trust - 
      ,47123 # Geneva International Centre for Humanitarian Demining -
      ,47137 # African Risk Capacity Group - 
      ,47502 # Global Fund for Disaster Risk Reduction
    )
  )
] = T

# Include all the financing under purpose codes
crs$`Crisis finance identified`[
  which(
    crs$purpose_code %in% c(
      12264 # COVID-19 control - 
      ,15220 # Civilian peace-building, conflict prevention and resolution - 
      ,15240 # Reintegration and SALW control - 
      ,15250 # Removal of land mines and explosive remnants of war - ,
      ,15261 # Child soldiers (prevention and demobilisation) - 
      ,43060 # Disaster Risk Reduction
    )
  )
] = T

# Run across the following purpose codes
crs$`Crisis finance eligible` = F

crs$`Crisis finance eligible`[
  which(
    crs$purpose_code %in% c(
      111 # Education, Level Unspecified - 
      ,11110 # Education policy and administrative management - 
      ,11120 # Education facilities and training - 
      ,11130 # Teacher training - 
      ,11182 # Educational research - 
      ,112 # Basic Education - 
      ,11220 # Primary education - 
      ,11230 # Basic life skills for adults - 
      ,11231 # Basic life skills for youth -
      ,11232 # Primary education equivalent for adults - 
      ,11240 # Early childhood education - 
      ,11250 # School feeding -
      ,11260 # Lower secondary education - 
      ,121 # Health, General - 
      ,12110 # Health policy and administrative management - 
      ,12191 # Medical services - 
      ,122 # Basic Health - 
      ,12220 # Basic health care - 
      ,12230 # Basic health infrastructure - 
      ,12240 # Basic nutrition - 
      ,12250 # Infectious disease control - 
      ,12261 # Health education - 
      ,12262 # Malaria control - 
      ,12263 # Tuberculosis control - 
      ,12281 # Health personnel development - 
      ,130 # Population Policies/Programmes & Reproductive Health - 
      ,13010 # Population policy and administrative management - 
      ,13020 # Reproductive health care - 
      ,13030 # Family planning - 
      ,13081 # Personnel development for population and reproductive health - 
      ,140 # Water Supply & Sanitation - 
      ,14010 # Water sector policy and administrative management - 
      ,14015 # Water sources conservation (including data collection) - 
      ,14020 # Water supply and sanitation – large systems - 
      ,14021 # Water supply – large systems - 
      ,14022 # Sanitation – large systems - 
      ,14030 # Basic drinking water supply and basic sanitation - 
      ,14031 # Basic drinking water supply -
      ,14032 # Basic sanitation - 
      ,14040 # River basins development - 
      ,14050 # Waste management/disposal - 
      ,14081 # Education and training in water supply and sanitation - 
      ,151 # Government & Civil Society-general - 
      ,15110 # Public sector policy and administrative management - 
      ,15111 # Public finance management (PFM) - 
      ,15114 # Domestic revenue mobilisation - 
      ,15142 # Macroeconomic policy - 
      ,15160 # Human rights - 
      ,15170 # Women’s rights organisations and movements, and government institutions - 
      ,15190 # Facilitation of orderly, safe, regular and responsible migration and mobility - 
      ,152 # Conflict, Peace & Security - 
      ,160 # Other Social Infrastructure & Services - 
      ,16010 # Social Protection - 
      ,16020 # Employment creation - 
      ,16050 # Multisector aid for basic social services - 
      ,16062 # Statistical capacity building - 
      ,210 # Transport & Storage - 
      ,21010 # Transport policy and administrative management - 
      ,21020 # Road transport - 
      ,21030 # Rail transport - 
      ,21040 # Water transport - 
      ,21050 # Air transport - 
      ,21061 # Storage - 
      ,21081 # Education and training in transport and storage - 
      ,240 # Banking & Financial Services - 
      ,24010 # Financial policy and administrative management - 
      ,24020 # Monetary institutions - 
      ,24030 # Formal sector financial intermediaries - 
      ,24040 # Informal/semi-formal financial intermediaries - 
      ,24050 # Remittance facilitation, promotion and optimisation - 
      ,24081 # Education/training in banking and financial services - 
      ,311 # Agriculture - 
      ,31110 # Agricultural policy and administrative management - 
      ,31120 # Agricultural development - 
      ,31130 # Agricultural land resources - 
      ,31140 # Agricultural water resources - 
      ,31191 # Agricultural services - 
      ,321 # Industry - 
      ,32130 # Small and medium-sized enterprises (SME) development - 
      ,410 # General Environment Protection - 
      ,41010 # Environmental policy and administrative management - 
      ,430 # Other Multisector - 
      ,43010 # Multisector aid - 
      ,43030 # Urban development and management - 
      ,43040 # Rural development - 
      ,43071 # Food security policy and administrative management - 
      ,43072 # Household food security programmes - 
      ,43082 # Research/scientific institutions - 
      ,510 # General budget support - 
      ,51010 # General budget support-related aid -
      ,520 # Development Food Assistance - 
      ,52010 # Food assistance - 
      ,600 # Action Relating to Debt - 
      ,60010 # Action relating to debt - 
      ,60020 # Debt forgiveness - 
      ,60030 # Relief of multilateral debt - 
      ,60040 # Rescheduling and refinancing - 
      ,60061 # Debt for development swap - 
      ,60062 # Other debt swap - 
      ,60063 # Debt buy-back
    )
  )
] = T

# crs = subset(crs, `Crisis finance identified` | `Crisis finance eligible`)

textual_cols_for_classification = c(
  "project_title",
  "short_description",
  "long_description"
)

crs = crs %>%
  unite(text, all_of(textual_cols_for_classification), sep=" ", na.rm=T, remove=F)

crs$text = collapse_whitespace(remove_punct(tolower(trimws(crs$text))))

# Exclude transactions that contain the following keywords from the data:
# Note, only exclude these for matching "relief" falsely
bad_relief_keywords = c(
  "Comic Relief"
  ,"Sport Relief"
  ,"Medical Relief Society"
  ,"Assemblies of God Relief and Development Services"
  ,"Catholic Relief Services"
  ,"AIDS Relief"
  ,"World Bicycle Relief"
  ,"KSrelief"
  ,"The RELIEF Centre"
  ,"Relief International"
)
bad_relief_keywords = collapse_whitespace(remove_punct(tolower(trimws(bad_relief_keywords))))
bad_relief_regex = paste0(
  "\\b",
  paste(bad_relief_keywords, collapse="\\b|\\b"),
  "\\b"
)
crs$`Contains bad relief` = grepl(bad_relief_regex, crs$text, perl=T, ignore.case = T)

# Manually check transactions that contain ‘debt relief’
crs$`Contains debt relief` = grepl("\\bdebt relief\\b", crs$text, perl=T, ignore.case=T)

crs$`Contains relief` = grepl("\\brelief\\b", crs$text, perl=T, ignore.case=T)


keywords = fread("data/keywords.csv")
keywords$keyword = quotemeta(collapse_whitespace(remove_punct(tolower(trimws(keywords$keyword)))))
# Exclude relief for now because extra logic for bad relief and debt relief
keywords = subset(keywords, keyword != "relief")

cf_keywords = subset(keywords, category=="CF")$keyword
cf_regex = paste0(
  "\\b",
  paste(cf_keywords, collapse="\\b|\\b"),
  "\\b"
)
paf_keywords = subset(keywords, category!="CF")$keyword
paf_regex = paste0(
  "\\b",
  paste(paf_keywords, collapse="\\b|\\b"),
  "\\b"
)
aa_keywords = subset(keywords, !category %in% c("CF", "PAF"))$keyword
aa_regex = paste0(
  "\\b",
  paste(aa_keywords, collapse="\\b|\\b"),
  "\\b"
)

crs$`Crisis finance keyword match` = grepl(cf_regex, crs$text, perl=T, ignore.case = T)
crs$`Crisis finance keyword match`[which(
  crs$`Crisis finance keyword match` == F & # If there are no other matches
    crs$`Contains relief` == T & # Except for "relief"
    crs$`Contains bad relief` == F & # And it's not a match for the bad ones
    crs$`Contains debt relief` == F # or debt relief
)] = T # Mark it true

# Capture activities which only match for debt relief and no other keywords
crs$`Only debt relief` = crs$`Contains debt relief` == T &
  crs$`Crisis finance keyword match` == F
# if these are later determined to be CF, retroactively mark "Review"
# but for now mark them as match
crs$`Crisis finance keyword match`[which(crs$`Contains debt relief`)] = T

crs$`PAF keyword match` = grepl(paf_regex, crs$text, perl=T, ignore.case = T)
crs$`AA keyword match` = grepl(aa_regex, crs$text, perl=T, ignore.case = T)

# Use ML to reduce review
crs$`Crisis finance determination` =
  ifelse(
    crs$`Crisis finance identified` | crs$`Crisis finance eligible`, # Must be id'd or eligible
         ifelse(
           crs$`Crisis finance keyword match`, # Must be keyword match
           ifelse(crs$`Crisis finance keyword match` & crs$`Crisis finance predicted ML`, "Yes", "Review"), # Yes if kw match and ML match, else review
           "No" # No if not keyword match
         ),
       "No" # No if not id'd or eligible
       )
crs$`Crisis finance determination`[which(crs$`Crisis finance identified`)] = "Yes"
# Mark those that passed other criteria but only for matching debt relief as review
crs$`Crisis finance determination`[which(
  crs$`Crisis finance determination` == "Yes" &
    crs$`Only debt relief` == T
  )] = "Review"

crs$`PAF determination` = ifelse(
  crs$`PAF keyword match`,
  ifelse(crs$`PAF keyword match` & crs$`PAF predicted ML`, "Yes", "Review"),
  "No"
)

crs$`AA determination` = ifelse(
  crs$`AA keyword match`,
  ifelse(crs$`AA keyword match` & crs$`AA predicted ML`, "Yes", "Review"),
  "No"
)


# Interrelationships
crs$`PAF determination`[which(crs$`PAF determination` != "Yes" & crs$`AA determination` == "Yes")] = "Yes"
crs$`PAF determination`[which(crs$`PAF determination` != "Yes" & crs$`AA determination` == "Review")] = "Review"
crs$`Crisis finance determination`[which(crs$`Crisis finance determination` != "Yes" & crs$`PAF determination` == "Yes")] = "Yes"
crs$`Crisis finance determination`[which(crs$`Crisis finance determination` != "Yes" & crs$`PAF determination` == "Review")] = "Review"
crs$`Crisis finance determination`[which(crs$humanitarian)] = "Yes"
crs$humanitarian[which(!crs$humanitarian & crs$`AA determination` == "Yes")] = T

describe(crs$`Crisis finance determination`)
describe(crs$`PAF determination`)
describe(crs$`AA determination`)

predicted_cols = names(crs)[which(endsWith(names(crs), "predicted ML"))]
for(predicted_col in predicted_cols){
  crs[which(crs[,predicted_col]==F),predicted_col] = ""
}

search_cols = names(crs)[which(endsWith(names(crs), "match"))]
for(search_col in search_cols){
  crs[which(crs[,search_col]==F),search_col] = ""
}

other_bool_cols = c("humanitarian", "Crisis finance identified", "Crisis finance eligible", "Contains debt relief")
for(other_col in other_bool_cols){
  crs[which(crs[,other_col]==F),other_col] = ""
}

keep= c(original_names,
        "humanitarian",
        "Crisis finance identified",
        "Crisis finance eligible",
        "Crisis finance determination",
        "Crisis finance keyword match",
        "Crisis finance predicted ML",
        "Crisis finance confidence ML",
        "PAF determination",
        "PAF keyword match",
        "PAF predicted ML",
        "PAF confidence ML",
        "AA determination",
        "AA keyword match",
        "AA predicted ML",
        "AA confidence ML",
        "Direct predicted ML",
        "Direct confidence ML",
        "Indirect predicted ML",
        "Indirect confidence ML",
        "Part predicted ML",
        "Part confidence ML"
)

crs = crs[order(
  crs$`Crisis finance determination`=="No",
  crs$`Crisis finance determination`=="Review",
  crs$`Crisis finance determination`=="Yes",
  -crs$`Crisis finance confidence ML`
),keep]


fwrite(crs,
       paste0("crs_",YEAR,"_cdp_automated.csv"))
