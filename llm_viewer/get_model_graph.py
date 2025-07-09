from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import importlib
import os
from hardwares.hardware_params import hardware_params
from model_analyzer import ModelAnalyzer
from utils import str_number
import numpy as np
import re
from backend_settings import avaliable_model_ids_sources

config_cache = {}


def get_analyer(model_id, hardware, config_path) -> ModelAnalyzer:
    config = f"{model_id}_{hardware}_{config_path}"
    if config not in config_cache:
        config_cache[config] = ModelAnalyzer(
            model_id,
            hardware,
            config_path,
            source=avaliable_model_ids_sources[model_id]["source"],
        )
    return config_cache[config]


# def get_model_config(model_id,config_path):
#     if model_id not in config_cache:
#         model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
#         config = importlib.import_module(config_path.replace("/", ".").replace(".py", ""))
#         config_cache[model_id] = model_config,config
#     return config_cache[model_id]


def get_quant_bit(dtype):
    if dtype == "FP16":
        return 16
    elif dtype == "INT8":
        return 8
    elif dtype == "INT4":
        return 4
    elif "bit" in dtype:
        bitwidth = int(re.findall(r"\d+", dtype)[0])
        return bitwidth
    else:
        raise ValueError(f"Unsupported dtype:{dtype}")


def get_model_graph(model_id, hardware, config_path, inference_config):

    # Roofline model
    w_bit = get_quant_bit(inference_config["w_quant"])
    a_bit = get_quant_bit(inference_config["a_quant"])
    kv_bit = get_quant_bit(inference_config["kv_quant"])
    seq_length = int(inference_config["seq_length"])
    batch_size = int(inference_config["batch_size"])
    use_flashattention = bool(inference_config["use_flashattention"])
    gen_length = int(inference_config["gen_length"])
    tp_size = int(inference_config["tp_size"])

    analyzer = get_analyer(model_id, hardware, config_path)
    result = analyzer.analyze(
        seqlen=seq_length,
        batchsize=batch_size,
        w_bit=w_bit,
        a_bit=a_bit,
        kv_bit=kv_bit,
        use_flashattention=use_flashattention,
        tp_size=tp_size
    )
    bandwidth, max_OPS, onchip_buffer = analyzer.get_hardware_info()
    GQA = analyzer.get_model_info()["GQA"]
    hardware_info = {
        "bandwidth": bandwidth,
        "max_OPS": max_OPS,
        "onchip_buffer": onchip_buffer,
    }

    nodes = [
        {
            "label": "input",
            "id": "input",
        }
    ]
    edges = []

    def write_to_node(name, OPs, memory_access, info, input_names=[]):
        node = {
            "label": name,
            "id": name,
            "description": f"OPs:{str_number(OPs)}, Access:{str_number(memory_access)}",
            "info": info,
        }
        if GQA and name in ["qk_matmul", "sv_matmul"]:
            node["label"] += "(GQA)"
        nodes.append(node)
        for input_name in input_names:
            edge = {"source": input_name, "target": name}
            edges.append(edge)

    if use_flashattention:
        layer_graph = analyzer.config.flashattention_transformer_layer_graph
    else:
        layer_graph = analyzer.config.transformer_layer_graph
    stage = inference_config["stage"]
    total_results = result["total_results"]
    if stage != "chat":
        result = result[stage]
    else:
        result = result["prefill"]

    for name, input_names in layer_graph.items():
        if name in ["input", "output"]:
            OPs = 0
            memory_access = 0
            info = {}
        else:
            OPs = result[name]["OPs"]
            memory_access = result[name]["memory_access"]
            info = result[name]
        write_to_node(name, OPs, memory_access, info, input_names)
    if stage == "chat":
        # seq_length:seq_length+gen_length
        total_results["chat"] = total_results["prefill"]
        n_divide = min(10, gen_length)
        for lengthi in np.linspace(seq_length + 1, seq_length + gen_length, n_divide):
            gen_result = analyzer.analyze(
                seqlen=lengthi,
                batchsize=batch_size,
                w_bit=w_bit,
                a_bit=a_bit,
                kv_bit=kv_bit,
                use_flashattention=use_flashattention,
            )
            for k, v in gen_result["total_results"]["decode"].items():
                total_results["chat"][k] += v * gen_length / n_divide
            for name, input_names in layer_graph.items():
                if name in gen_result["decode"]:
                    result[name]["OPs"] += (
                        gen_result["decode"][name]["OPs"] * gen_length / n_divide
                    )
                    result[name]["memory_access"] += (
                        gen_result["decode"][name]["memory_access"]
                        * gen_length
                        / n_divide
                    )
        for name, input_names in layer_graph.items():
            if name in ["input", "output"]:
                OPs = 0
                memory_access = 0
                info = {}
            else:
                OPs = result[name]["OPs"]
                memory_access = result[name]["memory_access"]
                info = {}
            write_to_node(name, OPs, memory_access, info, input_names)
    return nodes, edges, total_results, hardware_info
