# Mixture-of-Expert Classifier

Mixture-of-Expert PoC for a simple sequence classification task.

Unlike references where MoE is applied to MLP in transformer blocks, classification head is the target of MoE in this repository.

First `roberta-base` extracts features of sentence from `[CLS]` representation, mapped by a pooler layer.
This representation is then fed into either dense 2-layer MLP or its MoE counterpart to see effectiveness of MoE architecture for simple classification tasks.

Note that this MoE implementations are not optimized (using iteration to map each sample to experts), so it is slightly slower than its dense counterpart.

## TEST 1

- Experts : 2-layer MLP with GELU and Softmax activation.
- Method 
    - Each `[CLS]` token is routed to top-1 expert, and its probability is computed by a simple linear router.
    - Router-z-loss and load balancing loss are utilized. 
- For detail, see `MoE` class in `models.py`
- You can see [wandb log](https://wandb.ai/lucas01/moe_classifier?workspace=user-lucas01) here!


## TEST 2 (WIP)

- Experts : first linear layer within TEST 1 experts.
- Method
    - Same as TEST 1 except that second MLP layer is shared across experts.

## References

- [ST-MOE](https://arxiv.org/pdf/2202.08906.pdf)
- [Switch Transformer](https://arxiv.org/pdf/2101.03961.pdf)