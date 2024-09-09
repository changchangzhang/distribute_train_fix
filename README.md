# distribute_train_fix
author changchangzhang
date 20240906
To enable the polymorphic output shape support for collective ops, this change disables caching output_shape for collective ops. 
Every time an allGather collective op is called, a new output_shape will be inferred and assigned to the latest instance.shape.
However, The cost of fix code of collective_ops.cc is big, By fixing collective_all_gather method of cross_device_utils.py,aim to 
same gadients shape of each batch during traing.
