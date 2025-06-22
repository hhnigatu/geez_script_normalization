import pandas as pd
import sacrebleu
import re
from collections import Counter
from tqdm import tqdm

def normalize_amharic_H(token):

        # Normalizing token to the most frequent 
        rep1=re.sub('[ሃኀኃሐሓኻዃ]','ሀ',token)
        rep2=re.sub('[ሑኁኹዅ]','ሁ',rep1)
        rep3=re.sub('[ኂሒኺ]','ሂ',rep2)
        rep4=re.sub('[ኄኌሔኼዄ]','ሄ',rep3)
        rep5=re.sub('[ሕኅኽ]','ህ',rep4)
        rep6=re.sub('[ኆሖኾ]','ሆ',rep5)
        rep7=re.sub('[ሠ]','ሰ',rep6)
        rep8=re.sub('[ሡ]','ሱ',rep7)
        rep9=re.sub('[ሢ]','ሲ',rep8)
        rep10=re.sub('[ሣ]','ሳ',rep9)
        rep11=re.sub('[ሤ]','ሴ',rep10)
        rep12=re.sub('[ሥ]','ስ',rep11)
        rep13=re.sub('[ሦ]','ሶ',rep12)
        rep14=re.sub('[ዓኣዐ]','አ',rep13)
        rep15=re.sub('[ዑ]','ኡ',rep14)
        rep16=re.sub('[ዒ]','ኢ',rep15)
        rep17=re.sub('[ዔ]','ኤ',rep16)
        rep18=re.sub('[ዕ]','እ',rep17)
        rep19=re.sub('[ዖ]','ኦ',rep18)
        rep20=re.sub('[ፀ]','ጸ',rep19)
        rep21=re.sub('[ፁ]','ጹ',rep20)
        rep22=re.sub('[ፂ]','ጺ',rep21)
        rep23=re.sub('[ፃ]','ጻ',rep22)
        rep24=re.sub('[ፄ]','ጼ',rep23)
        rep25=re.sub('[ፅ]','ጽ',rep24)
        rep26=re.sub('[ፆ]','ጾ',rep25)
        #Normalizing words with Labialized Amharic characters such as በልቱዋል or  በልቱአል to  በልቷል  
        rep27=re.sub('[ቊ]','ቁ',rep25) #ቁ can be written as ቊ
        rep28=re.sub('[ኵ]','ኩ',rep27) #ኩ can be also written as ኵ  
        return rep28
    
def normalize_amharic_HSL(norm):
        norm = norm.replace("ሃ", "ሀ")
        norm = norm.replace("ሐ", "ሀ")
        norm = norm.replace("ሓ", "ሀ")
        norm = norm.replace("ኅ", "ሀ")
        norm = norm.replace("ኻ", "ሀ")
        norm = norm.replace("ኃ", "ሀ")
        norm = norm.replace("ዅ", "ሁ")
        norm = norm.replace("ሗ", "ኋ")
        norm = norm.replace("ኁ", "ሁ")
        norm = norm.replace("ኂ", "ሂ")
        norm = norm.replace("ኄ", "ሄ")
        norm = norm.replace("ዄ", "ሄ")
        norm = norm.replace("ኅ", "ህ")
        norm = norm.replace("ኆ", "ሆ")
        norm = norm.replace("ሑ", "ሁ")
        norm = norm.replace("ሒ", "ሂ")
        norm = norm.replace("ሔ", "ሄ")
        norm = norm.replace("ሕ", "ህ")
        norm = norm.replace("ሖ", "ሆ")
        norm = norm.replace("ኾ", "ሆ")
        norm = norm.replace("ሠ", "ሰ")
        norm = norm.replace("ሡ", "ሱ")
        norm = norm.replace("ሢ", "ሲ")
        norm = norm.replace("ሣ", "ሳ")
        norm = norm.replace("ሤ", "ሴ")
        norm = norm.replace("ሥ", "ስ")
        norm = norm.replace("ሦ", "ሶ")
        norm = norm.replace("ሼ", "ሸ")
        norm = norm.replace("ቼ", "ቸ")
        norm = norm.replace("ዬ", "የ")
        norm = norm.replace("ዲ", "ድ")
        norm = norm.replace("ጄ", "ጀ")
        norm = norm.replace("ጸ", "ፀ")
        norm = norm.replace("ጹ", "ፁ")
        norm = norm.replace("ጺ", "ፂ")
        norm = norm.replace("ጻ", "ፃ")
        norm = norm.replace("ጼ", "ፄ")
        norm = norm.replace("ጽ", "ፅ")
        norm = norm.replace("ጾ", "ፆ")
   
        norm = norm.replace("ዓ", "አ")
        norm = norm.replace("ዑ", "ኡ")
        norm = norm.replace("ዒ", "ኢ")
        norm = norm.replace("ዐ", "አ")
        norm = norm.replace("ኣ", "አ")
        norm = norm.replace("ዔ", "ኤ")
        norm = norm.replace("ዕ", "እ")
        norm = norm.replace("ዖ", "ኦ")
        
        norm=re.sub('(ሉ[ዋአ])','ሏ',norm)
        norm=re.sub('(ሙ[ዋአ])','ሟ',norm)
        norm=re.sub('(ቱ[ዋአ])','ቷ',norm)
        norm=re.sub('(ሩ[ዋአ])','ሯ',norm)
        norm=re.sub('(ሱ[ዋአ])','ሷ',norm)
        norm=re.sub('(ሹ[ዋአ])','ሿ',norm)
        norm=re.sub('(ቁ[ዋአ])','ቋ',norm)
        norm=re.sub('(ቡ[ዋአ])','ቧ',norm)
        norm=re.sub('(ቹ[ዋአ])','ቿ',norm)
        norm=re.sub('(ሁ[ዋአ])','ኋ',norm)
        norm=re.sub('(ኑ[ዋአ])','ኗ',norm)
        norm=re.sub('(ኙ[ዋአ])','ኟ',norm)
        norm=re.sub('(ኩ[ዋአ])','ኳ',norm)
        norm=re.sub('(ዙ[ዋአ])','ዟ',norm)
        norm=re.sub('(ጉ[ዋአ])','ጓ',norm)
        norm=re.sub('(ደ[ዋአ])','ዷ',norm)
        norm=re.sub('(ጡ[ዋአ])','ጧ',norm)
        norm=re.sub('(ጩ[ዋአ])','ጯ',norm)
        norm=re.sub('(ጹ[ዋአ])','ጿ',norm)
        norm=re.sub('(ፉ[ዋአ])','ፏ',norm)
        norm=re.sub('[ቊ]','ቁ',norm) 
        norm=re.sub('[ኵ]','ኩ',norm)
        norm=re.sub('\s+',' ',norm)
        
        return norm

def normalize_tigrinya(token):
        # Normalizing token to the most frequent 
        rep1=re.sub('[ኀ]','ሀ',token)
        rep2=re.sub('[ኁ]','ሁ',rep1)
        rep3=re.sub('[ኂ]','ሂ',rep2)
        rep4=re.sub('[ኃ]','ሃ',rep3)
        rep5=re.sub('[ኄኌ]','ሄ',rep4)
        rep6=re.sub('[ኅ]','ህ',rep5)
        rep7=re.sub('[ኆ]','ሆ',rep6)
        rep8=re.sub('[ሠ]','ሰ',rep7)
        rep9=re.sub('[ሡ]','ሱ',rep8)
        rep10=re.sub('[ሢ]','ሲ',rep9)
        rep11=re.sub('[ሣ]','ሳ',rep10)
        rep12=re.sub('[ሤ]','ሴ',rep11)
        rep13=re.sub('[ሥ]','ስ',rep12)
        rep14=re.sub('[ሦ]','ሶ',rep13)
        rep15=re.sub('[ፀ]','ጸ',rep14)
        rep16=re.sub('[ፁ]','ጹ',rep15)
        rep17=re.sub('[ፂ]','ጺ',rep16)
        rep18=re.sub('[ፃ]','ጻ',rep17)
        rep19=re.sub('[ፄ]','ጼ',rep18)
        rep20=re.sub('[ፅ]','ጽ',rep19)
        rep21=re.sub('[ፆ]','ጾ',rep20)
        #Normalizing words with Labialized Amharic characters such as በልቱዋል or  በልቱአል to  በልቷል  
        rep22=re.sub('[ቊ]','ቁ',rep21)
        rep23=re.sub('[ኵ]','ኩ',rep22) 
        return rep23

def get_translations(path, target_lang, normalize="None"):
    
    data=pd.read_csv(path)
    print(data.head(10))
    if target_lang not in data.columns or 'translated' not in data.columns:
        raise ValueError("The CSV file must contain '" + target_lang+ "' and 'translated' columns.")
    data[target_lang] = data[target_lang].fillna('').astype(str)
    data['translated'] = data['translated'].fillna('').astype(str)
    references = data[target_lang].tolist()
    references=[re.sub(r'[^\w\s0-9]', ' ', x).strip() for x in references]
    
    translations = data['translated'].tolist()
    translations=[re.sub(r'[^\w\s0-9]', ' ', x).strip() for x in  translations]
    

    
    if normalize=="H":
        if target_lang=='amh':
            references=[normalize_amharic_H(sentence) for sentence in references]
            translations=[normalize_amharic_H(sentence) for sentence in translations]
        elif target_lang=='tir':
            references=[normalize_tigrinya(sentence) for sentence in references]
            translations=[normalize_tigrinya(sentence) for sentence in translations]
    if normalize=="HSL":
        if target_lang=='amh':
            references=[normalize_amharic_HSL(sentence) for sentence in references]
            translations=[normalize_amharic_HSL(sentence) for sentence in translations]
        elif target_lang=='tir':
            print("HSL setting only for Amharic.")
    print(translations[1])
    print(references[1])
    return translations, references

file_path = ''  # Replace with path to file with reference and predictions.

translations, references=get_translations(file_path, "amh", "None")


print(len(translations))
print(len(references))

# BLEU score 
bleu = sacrebleu.corpus_bleu(translations, [references])
# chrF score 
chrf = sacrebleu.corpus_chrf(translations, [references])

print(f"BLEU Score: {bleu.score:.2f}")

print(f"chrF Score: {chrf.score:.2f}")