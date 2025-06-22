# A Case Against Implicit Standards: Homophone Normalization in Machine Translation for Languages that use the Ge’ez Script
[Paper]()

## Abstract
Homophone normalization–where characters that have the same sound in a writing script are mapped to one character–is a pre-processing step applied in Amharic Natural Language Processing (NLP) literature. While this may improve performance reported by automatic metrics, it also results in models that are not able to understand different forms of writing in a single language. Further, there might be impacts in transfer learning, where models trained on normalized data do not generalize well to other languages. In this paper, we experiment with monolingual training and cross-lingual transfer to understand the impacts of normalization on languages that use the Ge’ez script. We then propose a post-inference intervention in which normalization is applied to model predictions instead of training data. With our simple scheme of post-inference normalization, we show that we can achieve an increase in BLEU score of up to 1.03 while preserving language features in training. Our work contributes to the broader discussion on technology-facilitated language change and calls for more language-aware interventions.

## Data
* **Amharic**: Solomon Teferra Abate, Michael Melese, Martha Yifiru Tachbelie, Million Meshesha, Solomon Atinafu,
Wondwossen Mulugeta, Yaregal Assibie, Hafte Abera, Binyam Ephrem, Tewodros Abebe, Wondimagegnhue Tsegaye, Amanuel Lemma, Tsegaye Andargie, and Seifedin Shifaw. 2018. Parallel Corpora for bi-lingual English-Ethiopian Languages Statistical Machine Translation

* **Tigrinya**: Surafel M. Lakew, Matteo Negri, and Marco Turchi. 2020. Low Resource Neural Machine Translation:
A Benchmark for Five African Languages. arXiv preprint. ArXiv:2003.14402 AND Solomon Teferra Abate, Michael Melese, Martha Yifiru Tachbelie, Million Meshesha, Solomon Atinafu, Wondwossen Mulugeta, Yaregal Assibie, Hafte Abera, Binyam Ephrem, Tewodros Abebe, Wondimagegnhue Tsegaye, Amanuel Lemma, Tsegaye Andargie, and Seifedin Shifaw. 2018. Parallel Corpora for bi-lingual English-Ethiopian Languages Statistical Machine Translation
  
* **Ge'ez**: Henok Ademtew and Mikiyas Birbo. 2024. AGE: Amharic, Geez and English Parallel Dataset. In
Proceedings of the Seventh Workshop on Technologies for Machine Translation of Low-Resource Languages (LoResMT 2024)

