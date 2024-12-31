# Code of Fractal GNN

### 1. Requirements

```
transformers
dgl
numpy
torch
networkx
scikit-learn (sklearn)
```



### 2. How to Run

#### 2.1 Node Classification

```bash
python code_node_classification.py \
--seed 12306 \				# random seed
--data citeseer \			# datasets
--multivew \				# whether use multiview concatenation
--fractal \					# whether use fractal embedding
--fractal_concat concat\	# type of concatenation of fractal embedding, concat or sum
--epoch 30					# epochs of iteration
```

- `--multiview`为`store_true`参数，加入该参数表示在GAT的邻居表示基础上，拼接节点自身表示
- `--fractal`为`store_true`参数，加入该参数表示拼接上节点的分形表示
- 其余参数可见`code_node_classification.py`

#### 2.2 Graph Classification

目前基于GraphCL的对比学习过程进行分形相关修改，在Redditbinary上进行实验

```sh
# pretraining (contrastive learning)
python code_graphCL.py --config ./configs/pretrain_graphcl_GIN_redditbianary_{AUG_TYPE}.json
# finetune (graph classification)
python code_graphCL.py --config ./configs/finetune_graphcl_GIN_redditbianary_{AUG_TYPE}.json
```

- `AUG_TYPE`目前有3种选择：`dn`（drop node，随机去除节点），`renorm`（renormalization，基于分形性质聚类重构），`srw`（simple random walk，随机游走）
- 不通过`--config`传入配置时，可以通过指令传入`--epoch XXX`这些参数进行配置。

 

### 3. Logging

控制台logging的第一行会显示`Logging to {log_file_name}`，告知日志文件名

