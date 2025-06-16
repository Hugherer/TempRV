import models


def call(args):
    model = args.model
    if model == "code-davinci-002":
        res = models.code_davinci_002(args)
        return res
    if model == "text-bison@001":
        res = models.text_bison_001(args)
        return res
    if model == "code-bison@001":
        res = models.code_bison_001(args)
        return res
    if model == "text-davinci-003":
        res = models.text_davinci_003(args)
        return res
    if model == "code-davinci-edit-001":
        res = models.code_davinci_edit_001(args)
        return res
    if model == "bloom":
        res = models.bloom(args)
        return res
    if model == "gpt-3.5-turbo":
        res = models.gpt_35_turbo(args)
        return res
    if model == "gpt-4":
        res = models.gpt_4(args)
        return res
    if model == "bloomz":
        res = models.bloomz(args)
        return res
    if model == "deepseek_model":
        res = models.deepseek_model(args)
        return res
    if model == "llama_model":
        res = models.llama_model(args)
        return res
    if model == "llama3b_model":
        res = models.llama3b_model(args)
        return res
    if model == "qwen_model":
        res = models.qwen_model(args)
        return res
    if model == "qwen7b_model":
        res = models.qwen7b_model(args)
        return res
    raise Exception("Not a valid model.")

def FCQs_call(args):
    model = args.model
    if model == "code-davinci-002":
        res = models.code_davinci_002(args)
        return res
    if model == "text-bison@001":
        res = models.text_bison_001(args)
        return res
    if model == "code-bison@001":
        res = models.code_bison_001(args)
        return res
    if model == "text-davinci-003":
        res = models.text_davinci_003(args)
        return res
    if model == "code-davinci-edit-001":
        res = models.code_davinci_edit_001(args)
        return res
    if model == "bloom":
        res = models.bloom(args)
        return res
    if model == "gpt-3.5-turbo":
        res = models.gpt_35_turbo(args)
        return res
    if model == "gpt-4":
        res = models.gpt_4(args)
        return res
    if model == "bloomz":
        res = models.bloomz(args)
        return res
    if model == "deepseek_model":
        res = models.deepseek_model(args)
        return res
    if model == "llama_model":
        res = models.llama_model(args)
        return res
    if model == "llama3b_model":
        res = models.FCQs_llama3b_model(args)
        return res
    if model == "qwen_model":
        res = models.qwen_model(args)
        return res
    if model == "qwen7b_model":
        res = models.FCQs_qwen7b_model(args)
        return res
    raise Exception("Not a valid model.")
