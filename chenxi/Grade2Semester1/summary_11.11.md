## 数据增强对比实验
### 实验设置
- 基础模型：[LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch)
- 迭代次数: 100
- 数据集: Gowalla
- 评价指标: preicision, recall, ndcg
- 数据增强方式：节点合成
- 合成数量：每次迭代合成10个新节点
- 节点选择方式: 
  - **Random**: 随机选取2个物品节点.
  - **User relative**: 随机选取一个用户 $U_{a}$, 然后从与$U_{a}$嵌入的余弦相似度较高的其他用户中选取用户$U_{b}$，再分别从两个用户连接到的物品节点中各自选取一个
### 实验结果
|DataAugment|Precision|Recall|NDCG| Training time per epoch /s |
|----|----|----|----| ---- |
| None | 0.04490007 | 0.1499887 | 0.13220432 | 17.00 |
| Random | 0.04507 | 0.13508881 | 0.13113423 | 126.69 |
| User relative | 0.04490368 | **0.15129019** | **0.13483361** | 138.22 |

### 分析
- 根据相关用户选择物品节点进行合成效果比随机选取效果更好
- 目前还存在一些问题：
  - 性能提升不大
  - 训练时间过长
- 原因可能在于目前采用的是在训练到1/2后开始，每个epoch都进行一次增强，这可能会导致图频繁变动，模型难以收敛；并且训练时设置的迭代次数过少。可以考虑仅做一次增强，增加迭代次数

## 数据增强相关论文的表达和论述
### Improving Long-Tail Item Recommendation with Graph Augmentation
- 选自：CIKM2023
- 梳理思路
  - 现状：graph-based recommendation models have gained increasing popularity due to their ability to capture complex highorder user-item interactions（引出用GNN解决长尾问题）
  - 存在问题：
    - existing long-tail recommendation methods mainly focus on traditional neural networkbased models（现有方法仅关注网络结构）
    - it remains unexplored to solve the long-tail recommendation problem in graph-based models（前人没有探索过用解决长尾问题）
  - 现有方案及不足：
    - various approaches have been proposed at both **data** and algorithm levels（分为数据和算法层面的改进）
      - The sparse connectivity of tail items in the graph inhibits adequate information flow during the propagation phase in the graph neural network (GNN), limiting the potential for learning meaningful representations.（尾部项目连接少限制了GNN模型的信息传播能力）
      - The optimization loss during training tends to be overwhelmingly dictated by the more plentiful head nodes, marginalizing the tail nodes and exacerbating the skewed learning towards the head items.（训练损失由数据丰富的头部物品主导，尾部影响较小）
      - the imbalance between the head and tail items may lead to overfitting on the head items and poor generalization on the tail items.（头部易过拟合，尾部缺少泛化性能）
  - 提出自己的方案（长尾推荐图增强 GALORE）
    - To improve the connectivity of tail items, we incorporate item-to-item edge addition for tail items, allowing them to receive messages from nearby welllearned head items. （在头部和尾部物品之间建立边）
    - we propose a degree-aware edge dropping process, preserving more important edges for tail items and dropping unimportant edges for head items（根据度来选择删除的边，为尾部节点保留更多边，头部删除更多边）
    - we introduce a node synthesis method that synthesizes new data（提出了节点合成方法）

### Counterfactual Graph Augmentation for Consumer Unfairness Mitigation in Recommender Systems
- 选自：CIKM2023
- 梳理思路
  - 现状：Concerted efforts towards explaining unfairness in recommendation have been recently made（引出推荐公平性问题）
  - 存在问题：
    - none of them has led to a mitigation procedure that leverages the identified explanations to mitigate the measured unfairness（现有模型无法很好解决公平问题）
  - 现有方案及不足：
    - A first attempt to inform a mitigation procedure through explanation techniques was proposed by [12]. However, their method operates on Graph Neural Networks (GNNs) to detect the graph nodes affecting unfairness in classification tasks, limiting its adoption to networks of user-item interactions and to the recommendation task in general.（所指论文基于GNN，但局限于节点分类任务，缺乏普适性）
  - 提出自己的方案（反事实图增强）
    - we assume that the actions of the users in a demographic group led the model to advantage them.Thus, we hypothesize a counterfactual world where the disadvantaged users can benefit from new edges to improve their recommendation utility（假定弱势用户可以从额外的边中受益(为弱势用户添加额外的边)）

### LLMRec: Large Language Models with Graph Augmentation for Recommendation
- 选自：WSDM 2024
- 梳理思路&存在问题
  - 现状：In modern recommender systems, such as Netflix, the side information available exhibits heterogeneity. This diverse content offer distinct ways to characterize user preferences. However, despite significant progress, these methods often face challenges related to data scarcity and issues associated with handling side information.（引出利用额外信息和数据稀疏问题）
  - 现有方案不足：
    - GNN+CF：While many efforts (e.g., NGCF [41], LightCGN [11]) tried powerful graph neural networks(GNNs) in collaborative filtering(CF), they face limits due to insufficient supervised signals（缺少监督信号，因此需要引入其他模态的额外信息）
    - Recommender systems that incorporate side information often encounter significant issues that can negatively impact their performance. i) Data Noise; ii) Data heterogeneity; iii) Data incompleteness（额外信息存在噪声、异质和缺失问题）
  - 提出自己的方案（基于LLM的交互图增强，LLMRec）
    - LLMRec leverages large language models (LLMs) to predict user-item interactions from a natural language perspective（利用来自其他模态的额外信息预测交互，边添加）
    - The low-quality and incomplete side information is enhanced by leveraging the extensive knowledge of LLMs （大模型中丰富的知识完善了低质量和带缺失的额外信息，节点嵌入增强）