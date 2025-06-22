# A Case Against Implicit Standards: Homophone Normalization in Machine Translation for Languages that use the Ge’ez Script
[Paper]()

## Abstract
Homophone1 normalization–where characters that have the same sound in a writing script are mapped to one character–is a pre-processing step applied in Amharic Natural Language Processing (NLP) literature. While this may improve performance reported by automatic metrics, it also results in models that are not able to understand different forms of writing in a single language. Further, there might be impacts in transfer learning, where models trained on normalized data do not generalize well to other languages. In this paper, we experiment with monolingual training and cross-lingual transfer to understand the impacts of normalization on languages that use the Ge’ez script. We then propose a post-inference intervention in which normalization is applied to model predictions instead of training data. With our simple scheme of post-inference normalization, we show that we can achieve an increase in BLEU score of up to 1.03 while preserving language features in training. Our work contributes to the broader discussion on technology-facilitated language change and calls for more language-aware interventions.

## Data
* **Amharic**:

* **Tigrinya**:
  
* **Ge'ez**:

