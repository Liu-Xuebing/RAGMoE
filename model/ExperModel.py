import torch
import torch.nn as nn


# 定义专家模块，每个专家可以是一个简单的全连接层
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Expert, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x)


# 定义 MoE 层，包含门控机制和多个专家
class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])  #W_e
        self.gate = nn.Linear(input_dim, num_experts)  # gate decision, W_g


    def forward(self, x):
        gate_output = torch.softmax(self.gate(torch.cat([x],  dim=-1)), dim=-1)  # 生成每个专家的权重
        # _, top1_expert_idx = torch.topk(gate_output, k=1, dim=-1)
        # top1_expert_idx = top1_expert_idx.squeeze(dim=-1)
        # outputs = []
        # for ix, top1 in enumerate(top1_expert_idx):
        #     output = self.experts[top1](x[ix].unsqueeze(0))
        #     outputs.append(output)
        # return torch.cat(outputs, dim=0)

        ## soft-control
        output = 0
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            output += gate_output[:, i:i+1] * expert_output  # weighted the results of each expert
        return output


# 定义并行处理的函数
class ParallelFFNMoE(nn.Module):
    def __init__(self, ffn, moe, coe_lambda):
        super(ParallelFFNMoE, self).__init__()
        self.ffn = ffn
        self.moe = moe
        self.coe_lambda = coe_lambda

    def forward(self, x, id):
        # 创建一个空列表，用于存放每个token计算的结果
        final_output = torch.zeros_like(x)

        for i in range(x.size(1)):
            token = x[:, i, :]
            if i < id and x.size(1) != 1:
                token_ffn_output = self.ffn(token)  # FFN处理
                final_output[:, i:i+1, :] = token_ffn_output.unsqueeze(dim=1)
            else:
                token_ffn_output = self.ffn(token)  # FFN处理
                token_moe_output = self.moe(token)  # MoE处理
                final_output[:, i:i+1, :] = token_ffn_output.unsqueeze(dim=1) + (self.coe_lambda * token_moe_output.unsqueeze(dim=1))

        # for i in range(x.size(1)):
        #     token = x[:, i, :]
        #     token_moe_output = self.moe(token)  # MoE处理
            # print(token_moe_output.size())
        # token_output = token_ffn_output + self.coe_lambda * token_moe_output
        # token_outputs.append(token_output.unsqueeze(dim=1))
        # print(len(token_outputs))

        return final_output

