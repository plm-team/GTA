
def get_num_attention_heads(model_params):
    return getattr(model_params, "num_attention_heads")

def get_hidden_size(model_params):
    return getattr(model_params, "hidden_size")

def get_num_value_heads(model_params):
    return getattr(model_params, "num_value_heads")

def get_num_key_heads(model_params):
    return getattr(model_params, "num_key_heads")

def get_num_query_heads(model_params):
    return getattr(model_params, "num_query_heads")

def get_query_dim(model_params):
    return getattr(model_params, "query_dim")

def get_value_dim(model_params):
    return getattr(model_params, "value_dim")

def get_key_dim(model_params):
    return getattr(model_params, "key_dim")

def get_norm_layers(model_params):
    return ["attn_norm", "mlp_norm"]

def get_num_hidden_layers(model_params):
    return getattr(model_params, "num_hidden_layers")

def get_intermediate_size(model_params):
    return getattr(model_params, "intermediate_size")

def get_vocab_size(model_params):
    return getattr(model_params, "vocab_size")

def post_process(model_params,args):
    hiddensize=get_hidden_size(model_params)
    vocab_size=get_vocab_size(model_params)
    layers=[]
    for stage in ["prefill", "decode"]:
        layers.append({
            'name': 'lm_head',
            'stage':stage,
            'OPs':args['batchsize']*hiddensize*vocab_size*2,
            'load_weight':hiddensize*vocab_size *args['w_byte'],
            'load_act':hiddensize*args['a_byte'],
            'store_act':vocab_size*args['a_byte'],
        })
    return layers

def get_linear_layers(model_params, tp_size: int):
    hidden_size=get_hidden_size(model_params)
    intermediate_size=get_intermediate_size(model_params)
    key_heads=get_num_key_heads(model_params)
    value_heads=get_num_value_heads(model_params)
    query_heads=get_num_query_heads(model_params)

    key_dim = get_key_dim(model_params)
    value_dim = get_value_dim(model_params)
    query_dim = get_query_dim(model_params)
    
    return {
        "q_proj":[hidden_size, query_dim * query_heads],
        "k_proj":[hidden_size, key_dim * key_heads],
        "v_proj":[hidden_size, value_dim * value_heads],
        "out_proj":[hidden_size, hidden_size],
        "v2o_proj":[query_heads*value_dim, hidden_size//query_heads],
        "g_proj":[hidden_size, hidden_size],
        "gate_proj":[hidden_size, intermediate_size],
        "up_proj":[hidden_size,intermediate_size],
        "down_proj":[intermediate_size, hidden_size],
    }

# name, input_names
transformer_layer_graph={
    "input":[],
    "attn_norm": ["input"],
    "q_proj":["attn_norm"],
    "k_proj":["attn_norm"],
    "v_proj":["attn_norm"],
    "g_proj":["attn_norm"],
    "qk_matmul":["q_proj","k_proj"],
    "softmax":["qk_matmul"],
    "sv_matmul":["softmax","v_proj"],
    "v2o_proj":["sv_matmul"],
    "output_add":["sv_matmul", "v2o_proj"],
    "gate_act":["output_add", "g_proj"],
    "out_proj":["gate_act"],
    "attn_add":["input","out_proj"],
    "mlp_norm":["attn_add"],
    "gate_proj":["mlp_norm"],
    "up_proj":["mlp_norm"],
    "mlp_act":["up_proj","gate_proj"],
    "down_proj":["mlp_act"],
    "mlp_add":["attn_add","down_proj"],
    "output":["mlp_add"]
}

