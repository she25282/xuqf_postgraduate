# Methods in Graph Augment

## 特征增强
- 特征交换 
  ![alt text](image/image-9.png)
- 特征掩盖
  ![alt text](image/image-10.png)
## 节点增强
- 节点删除
  ![alt text](image/image-11.png)
- 节点合成
  ![alt text](image/image-12.png)
## 边增强
- 边删除
- 边增加
![alt text](image/image-13.png)
## 子图增强
- 子图采样
  ![alt text](image/image-14.png)
- 子图替换 wasd
  ![alt text](image/image-15.png)
## 全图增强
- 图聚合
  ![alt text](image/image-16.png)
- 图粗化/细化
  ![alt text](image/image-17.png)

## 个人想法
- 节点合成
  - 从完整数据中选取(头部+尾部)
  - 分别从尾部和头部数据中随机选取一对相似的用户节点表示
  - 分别从头部和尾部用户直接相连的物品节点中随机选取一个，作为合成的源节点
  - 合成的节点连接到尾部用户