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

```bash
python main_code.py \
--seed 12306 \				# random seed
--data citeseer \			# datasets
--multivew \				# whether use multiview concatenation
--fractal \					# whether use fractal embedding
--fractal_concat concat\	# type of concatenation of fractal embedding, concat or sum
--epoch 30					# epochs of iteration
```

- `--multiview`为`store_true`参数，加入该参数表示在GAT的邻居表示基础上，拼接节点自身表示
- `--fractal`为`store_true`参数，加入该参数表示拼接上节点的分形表示
- 其余参数可见`main_code.py`



### 3. Logging

输出日志见`./log`文件夹。输出日志文件名格式为`GAT_{method}_{data}.log`

控制台logging的第一行会显示`Logging to {log_file_name}`，告知日志文件名

