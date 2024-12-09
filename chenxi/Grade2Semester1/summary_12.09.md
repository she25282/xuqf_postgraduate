## 本周事项
- 数据增强组合实验
- 小论文修改
### 数据增强组合实验
- 实验目的：组合所提出的2种数据增强方法，确认同时使用时的效果
- 实验设置：
  - 数据集：Gowalla
  - 迭代次数：1000
  - 增强方式：仅在1/2 epoch处进行一次增强，执行200个节点合成/边添加
  - 实验组/对照组
    - 不使用数据增强（LightGCN）
    - 参考论文中的数据增强（GALORE）
    - 基于用户相似度的物品节点合成（sim_synth）
    - 基于用户相似度的边添加（sim_edge_add）
    - 随机物品节点合成+随机头尾物品节点边添加（random_synth+random_edge_add）
    - 基于用户相似度的物品节点合成+随机头尾物品节点边添加（sim_synth+random_edge_add）
    - 随机物品节点合成+基于用户相似度的头尾物品节点边添加（random_synth+sim_edge_add）
    - 基于用户相似度的物品节点合成+基于用户相似度的头尾物品节点边添加（sim_synth+sim_edge_add）
- 实验结果
  |DataAugment|Recall@20|NDCG@20| TCoverage@20|
  |----|----| ---- | ---- |
  | LightGCN | 0.1558 | 0.1316 | 3.1978 |
  | GALORE | 0.1654 | 0.1407 | ---- | 
  | sim_synth | 0.1801 | 0.1534 | 3.2356 |
  | sim_edge_add | **0.1812** | 0.1545 | 3.1978 | 
  |random_synth+random_edge_add | 0.1794 | 0.1525 | 3.0595 |
  |sim_synth+random_edge_add | 0.1801 | 0.1529 | 3.2287 |
  |random_synth+sim_edge_add | 0.1802 | 0.1536 | 3.2364 |
  |sim_synth+sim_edge_add | 0.1804 | **0.1545** | **3.2461** |
- 实验结论：
  - 两种增强方法结合相较单个方法在大部分指标上都更优，在召回率上有较小的差距，但这是由于推荐更多尾部物品导致的，是我们期待的结果
  - 基于相似度的方法比随机方法有着更好的效果，这也验证了我们的观点：针对推荐的图增强应基于高相关性的异质节点(物品的异质节点就是用户节点)
### 小论文修改
《》括起来的部分为新加的部分
```
Highly-skewed long-tail item distributions have been proved to undermine the 
performance of recommender systems, especially in dealing with tail items with little 
user feedback. Recent researches on Graph-based methods have shown their ability to 
effectively model user-item interactions. However, existing works still leave two 
challenges unsolved: (i) lack of user feedback for tail items, 《which could cause 
overfitting on tail recommendation》 and (ii) Trade-off between performance on tail 
items and that of all items.

To address these problems, we propose a novel Mixture of Experts with Graph 
Augmentation for Long Tail Recommendation (MegaLTR). Specifically, (i) We introduce a 
novel data augmentation method: pseudo-node synthesis and edge addition based on 
heterogeneous neighbor nodes 《that transfers knowledge from head item nodes to tail 
nodes and expands the tail data》 to address the issue of sparse interactions of tail 
items; (ii) we design a Mixture of Experts (MoE)-based contrastive distilling learning 
method that enables different MoEs to learn from tail distribution and overall 
distribution respectively, then distill the knowledge of them into a student MoE that 
could better balance the performance on head and tail items. Experiments on public 
datasets show that our proposed MegaLTR significantly outperforms strong baselines for 
tail items recommendation. The code is available on https://github.com/xxx/xxxx

高度倾斜的长尾物品分布已被证明会影响推荐系统的表现，尤其是在应对具仅有少量用户交互的尾部物品时。
最近的基于图的方法的研究展现了它们高效建模用户交互的能力。然而，现有的工作仍然遗留了2个未解决的
问题：1. 尾部项目缺少用户交互，《而这可能会导致尾部推荐时过拟合》；2. 如何权衡模型在头部和尾部数
据的上的表现。

为了解决这些问题，我们提出了一个全新的针对长尾推荐的图增强专家混合模型(MegaLTR)。具体来说，1. 
我们引入了一种新的数据增强方法：基于异质邻居节点的伪节点合成和边增加，《通过将头部物品的知识迁移
到尾部节点上并扩充尾部数据，以解决尾部交互稀疏的问题》；2. 我们设计了一个基于MoE的对比蒸馏学习方
法，它能让不同的MoE分别学习尾部分布和整体分布，然后将它们的知识转移到一个能更好平衡在头部和尾部
数据上表现的学生MoE中。在公开数据集上进行的实验结果表明我们提出的MegaLTR模型比其他优秀长尾推荐
模型有更好的表现。
```

总词数(英文)：205