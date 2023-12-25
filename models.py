
import torch.nn as nn
from torch.nn.functional import one_hot
import torch

class Dense(nn.Module):
    def __init__(self, roberta_model):
        super(Dense, self).__init__()
        self.feature_extractor = roberta_model
        hidden_size = self.feature_extractor.config.hidden_size
        for n,p in self.feature_extractor.named_parameters():
            if "pooler" not in n:
                p.requires_grad = False
        self.classifier = nn.Sequential(
                                        nn.Linear(hidden_size, hidden_size),
                                        nn.GELU(),
                                        nn.Linear(hidden_size, 2),
                                        nn.Softmax(dim=-1)
                                    )
        self.loss = nn.CrossEntropyLoss()


    def forward(self, **kwargs):
        class_label = kwargs.pop("class_label", None)
        out = self.feature_extractor(**kwargs)    
        out = out.pooler_output
        logits = self.classifier(out)
        loss = None
        pred = logits.max(dim=-1).indices
        if class_label is not None:
            ce_loss = self.loss(logits, class_label)
            loss = ce_loss
        return loss,  {"ce_loss": ce_loss, "pred": pred}
    
class MoE(nn.Module):
    def __init__(self, roberta_model, num_experts=8):
        super(MoE, self).__init__()
        self.feature_extractor = roberta_model
        hidden_size = self.feature_extractor.config.hidden_size
        self.num_experts = num_experts
        for n,p in self.feature_extractor.named_parameters():
            if "pooler" not in n:
                p.requires_grad = False
        self.gate = nn.Linear(hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([nn.Sequential(
                                                    nn.Linear(hidden_size, hidden_size),
                                                    nn.GELU(),
                                                    nn.Linear(hidden_size, 2),
                                                    nn.Softmax(dim=-1)
                                                ) for _ in range(num_experts)])
        self.loss = nn.CrossEntropyLoss()

    def forward(self, **kwargs):
        class_label = kwargs.pop("class_label", None)
        out = self.feature_extractor(**kwargs)    
        out = out.pooler_output
        logits = self.gate(out)

        # define balancing loss (`https://arxiv.org/pdf/2101.03961.pdf`)
        routing_prob = nn.functional.softmax(logits, dim=-1) # batch_size x num_experts
        dispatch_onehot = one_hot(routing_prob.max(dim=-1).indices, num_classes=self.num_experts) # batch_size x num_experts
        routing_prob_mean = routing_prob.mean(dim=0)
        dispatch_fraction = dispatch_onehot.float().mean(dim=0)
        balancing_loss = (routing_prob_mean * dispatch_fraction).sum()*self.num_experts 

        # define router_z_loss (`https://arxiv.org/pdf/2202.08906.pdf`)
        router_z_loss = logits.logsumexp(dim=-1).pow(2).mean() 
        
        
        logits = [self.experts[idx](_out[None,:]) for _out, idx in zip(out, routing_prob.max(dim=-1).indices)] 
        logits = torch.cat(logits, dim=0) # batch_size x 2
        weight = routing_prob.max(dim=-1).values + (1 - routing_prob.max(dim=-1).values).detach() # straight-through trick
        # weight = routing_prob.max(dim=-1).values
        weighted_logits = logits * weight[:,None]
        loss = None
        pred = weighted_logits.max(dim=-1).indices
        if class_label is not None:
            ce_loss = self.loss(weighted_logits, class_label)
            loss = (ce_loss + 0.01*balancing_loss + 0.001*router_z_loss) # using best coefficients from each paper

        return loss, {"ce_loss": ce_loss, "balancing_loss" : balancing_loss, "router_z_loss" : router_z_loss, "pred": pred}