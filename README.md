# Transformers Presentation - GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

## 1. Overview
The paper focuses on advancing the efficiency and effectiveness of Transformer models, particularly in the context of decoder inference speed and model quality. This research is grounded in addressing the computational intensity and memory bandwidth overhead that plagues autoregressive decoder inference in Transformer models. The paper introduces a novel methodology known as Grouped-Query Attention (GQA), which is an innovative take on optimizing model structure for better performance. 

## 2. Introduction

The research begins by identifying the challenge of memory bandwidth overhead in Transformer models during decoder inference, a critical bottleneck that slows down model performance. The conventional multi-query attention (MQA) mechanism, while speeding up inference by utilizing a single key-value head, often results in a trade-off with model quality. To address these limitations, the paper proposes a two-fold solution:

### 2.1. Uptraining existing models with MQA:
Leveraging only 5% of the original pre-training compute, this process efficiently transforms multi-head attention models into their MQA counterparts.
2: Introducing Grouped-Query Attention (GQA):
### 2.2. GQA is presented as a generalized form of MQA that employs an intermediate number of key-value heads, striking a balance between speed and quality.

## 3. Methodology

The methodology section delves deep into the procedural aspects of implementing GQA. It consists of two primary processes:

### 3.1 Uptraining: 

Transforming multi-head models into MQA models involves pooling the projection matrices of key and value heads into a single matrix, followed by additional pre-training to adapt the model to this new structure.

![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](https://github.com/FrankYang7777/GQA-Transformers-Presentation/blob/main/Overview%20Of%20Conversion%20from%20multi-head%20to%20multi-query%20attention.png)

### 3.2 Grouped-Query Attention: 

GQA divides query heads into groups, with each group sharing a single key and value head. This structure interpolates between the multi-head and MQA setups, aiming for a balance of speed and accuracy.

![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](https://github.com/FrankYang7777/GQA-Transformers-Presentation/blob/main/Overview%20of%20grouped-query%20method.png)

## 4. Experiments

### 4.1 Experimental Setup

### 4.2 Main Results

### 4.3 Ablations

## 5. Related Work

