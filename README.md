# Matrix

## English Data Clean Pipeline
The pipeline involves filtering and multi-step deduplication. The concrete use case for this pipeline is detailed in the README.md file located in the __english_data_clean_pipeline__ directory. The pipeline is illstruacted as follows.

<a name="english_data_pipeline"></a>
![english_data_pipeline](./english_data_clean_pipeline/english_data_pipeline.png)
### Filter
Given that open-source datasets often include low-quality data, we propose heuristic rules for web page filtering. These rules aim to identify and remove poor-quality data, thereby preventing potential degradation in model.  pre-training performance. Our cleaning strategies incorporate methodologies from diverse sources, including [RefinedWeb](https://arxiv.org/abs/2306.01116) and [CCNet](https://arxiv.org/abs/2311.01149). We also adopt some rules that are applied while training other models, such as [Gopher](https://arxiv.org/abs/2112.11446) and [T5](https://arxiv.org/abs/1910.10683).

### Deduplication
By eliminating duplicates, we can significantly reduce the rate of emitted memorizations and make model training more efficien. Repetitions can be categorized into exact duplicates and near duplicates. For exact duplicates, we employ exact document deduplication to remove them. For near duplicates, we utilize Minhash LSH deduplication to remove them as much as possible. In addition, there are instances where parts of the text
are completely duplicated, and in these cases, the Minhash method struggles to remove them. To address this, we have adopted two methods for partially removing such content: paragraph deduplication and exact substring deduplication. 


## Chinese Data Clean Pipeline
The pipeline involves filtering and multi-step deduplication. The concrete use case for this pipeline is detailed in the README.md file located in the __chinese_data_clean_pipeline__ directory. The pipeline is illstruacted as follows.

![chinses_data_pipeline](image.png)
### Filtering
The filtering rules for Chinese datasets are specifically tailored to address their unique challenges, differing from those applied to English datasets. Considering the large pro-portion of HTML-converted data in Chinese datasets, we focus intensively on eliminatingHTML-related artifacts and rectifying textual inconsistencies. Furthermore, given the signif-icant linguistic differences between Chinese and English, we conduct targeted sampling of documents within Chinese datasets, which aims to reassess and adjust the thresholds and details of our filtering rules, ensuring their suitability for the unique language characteristics of Chinese text. For example, we refine the rules to distinguish between ’characters’ and ’words’ in Chinese texts, adapting the tokenization method accordingly.

### Deduplication
The deduplication of Chinese data includes Exact Document Deduplication,MinHash Deduplication, and Similar Line Deduplication. 

## Get High quality Data
 Inspired by the procedure of collecting math data from [DeepSeekMath](https://arxiv.org/abs/2402.03300), we designed an iterative pipeline to acquire large-scale, high-quality data from specific fields using Common Crawl. These fields include math, K12, and wiki.The concrete use case for this module is detailed in the README.md file located in the __get_high_quality_data__ directory.

