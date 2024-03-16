# Transformers Presentation - GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

Presenter: Siyu Yang

## 1. Overview
The research team focuses on advancing the efficiency and effectiveness of Transformer models, particularly in the context of decoder inference speed and model quality. This research is grounded in addressing the computational intensity and memory bandwidth overhead that plagues autoregressive decoder inference in Transformer models. The paper introduces a novel methodology known as Grouped-Query Attention (GQA), which is an innovative take on optimizing model structure for better performance. 

## 2. Introduction

The research begins by identifying the challenge of memory bandwidth overhead in Transformer models during decoder inference, a critical bottleneck that slows down model performance. The conventional multi-query attention (MQA) mechanism, while speeding up inference by utilizing a single key-value head, often results in a trade-off with model quality. To address these limitations, the paper proposes a two-fold solution:

### 2.1. Uptraining existing models with MQA:
Leveraging only 5% of the original pre-training compute, this process efficiently transforms multi-head attention models into their MQA counterparts.
![](https://github.com/FrankYang7777/GQA-Transformers-Presentation/blob/main/Overview%20Of%20Conversion%20from%20multi-head%20to%20multi-query%20attention.png)

### 2.2. Introducing Grouped-Query Attention (GQA)
GQA is presented as a generalized form of MQA that employs an intermediate number of key-value heads, striking a balance between speed and quality.
![](https://github.com/FrankYang7777/GQA-Transformers-Presentation/blob/main/Overview%20of%20grouped-query%20method.png)

## 3. Methodology

The methodology section delves deep into the procedural aspects of implementing GQA. It consists of two primary processes:

### 3.1 Uptraining: 

Transforming multi-head models into MQA models involves pooling the projection matrices of key and value heads into a single matrix, followed by additional pre-training to adapt the model to this new structure.

<img width="490" alt="Screenshot 2024-03-16 150203" src="https://github.com/FrankYang7777/GQA-Transformers-Presentation/assets/142248146/184be52d-f838-4350-9f95-424a1afac17b">

### 3.2 Grouped-Query Attention: 

GQA divides query heads into groups, with each group sharing a single key and value head. This structure interpolates between the multi-head and MQA setups, aiming for a balance of speed and accuracy.

<img width="490" alt="Screenshot 2024-03-16 151237" src="https://github.com/FrankYang7777/GQA-Transformers-Presentation/assets/142248146/6834293e-8ee1-401d-988a-4c86fcb4ced4">

## 4. Experiments

### 4.1 Experimental Setup
The models evaluated include variations of the T5 architecture, specifically the T5 Large and XXL models with MHA, as well as versions of these models uptrained to utilize MQA and GQA. These models were tested across a diverse set of tasks including summarization, translation, and question answering to assess their general applicability and performance.

Configurations: The experiments leveraged the T5.1.1 architecture implemented with JAX, Flax, and Flaxformer, evaluating both T5 Large and XXL configurations.

Uptraining: Models were initialized from public T5.1.1 checkpoints and uptrained to use MQA or GQA structures. A key part of the process was the mean pooling of key and value heads into a single head for MQA, and into grouped heads for GQA, followed by additional pre-training using a fraction (5%) of the original pre-training compute.

Data and Tasks: The datasets used for evaluation included CNN/Daily Mail, arXiv, PubMed, MediaSum, Multi-News for summarization, WMT 2014 English-to-German for translation, and TriviaQA for question answering. These datasets were chosen to cover a range of input lengths and complexities.


### 4.2 Main Results

The experiments demonstrated several key findings:

Performance and Inference Time: GQA models achieved a close approximation to the quality of MHA models while offering inference speeds comparable to MQA models. This was evident across all tested datasets, showing GQA's effectiveness in providing a balanced approach between speed and accuracy.

Fine-tuning and Inference Details: For fine-tuning, consistent settings were used across all tasks, including learning rate, batch size, and dropout rate. The models were trained until convergence, and greedy decoding was used for inference.

### 4.3 Ablations

Checkpoint Conversion Methods: Different strategies for converting MHA models to MQA were tested, including mean pooling, selecting the first head, and random initialization. Mean pooling emerged as the most effective method, likely due to its ability to preserve information from the pre-trained model.

Uptraining Proportion: The effect of different uptraining proportions on the performance of T5 XXL with MQA and GQA was examined. It was found that GQA achieved reasonable performance immediately after conversion, while MQA required uptraining to reach useful levels of performance.

Number of GQA Groups: The influence of the number of groups in GQA on inference speed was analyzed. The study suggested a trade-off where increasing the number of groups from MQA results in modest slowdowns initially, with higher costs as the configuration moves closer to full MHA. An optimal middle ground was identified.

