import os
from statistics import mode

import openai
from openai import OpenAI
import deepseek
import requests
import vertexai
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
from vertexai.preview.language_models import TextGenerationModel, CodeGenerationModel
from http import HTTPStatus
import dashscope

import prompting


def gpt_35_turbo(args):
    if args.keyfile != "":
        keyfile = args.keyfile
    else:
        keyfile = os.path.join(args.keydir, "oai_key.txt")
    key = open(keyfile).readline().strip("\n")
    if key == "":
        raise Exception("No key provided.")
    openai.api_key = key
    if args.num_tries == "":
        n = 3
    else:
        n = int(args.num_tries)
        if n > 5:
            n = 5
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompting.prompt(args)}],
        n=n,
        temperature=args.temperature,
        stop="FINISH",
    )
    choices = []
    for i in range(0, n):
        output = response["choices"][i]["message"]["content"]
        print("OUTPUT")
        print(output)
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)


def gpt_4(args):
    if args.keyfile != "":
        keyfile = args.keyfile
    else:
        keyfile = os.path.join(args.keydir, "oai_key.txt")
    key = open(keyfile).readline().strip("\n")
    if key == "":
        raise Exception("No key provided.")
    openai.api_key = key
    if args.num_tries == "":
        n = 3
    else:
        n = int(args.num_tries)
        if n > 5:
            n = 5
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompting.prompt(args)}],
        n=n,
        temperature=args.temperature,
        stop="FINISH",
    )
    choices = []
    for i in range(0, n):
        output = response["choices"][i]["message"]["content"]
        print("OUTPUT")
        print(output)
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)


def code_davinci_002(args):
    if args.keyfile != "":
        keyfile = args.keyfile
    else:
        keyfile = os.path.join(args.keydir, "oai_key.txt")
    key = open(keyfile).readline().strip("\n")
    if key == "":
        raise Exception("No key provided.")
    openai.api_key = key
    if args.num_tries == "":
        n = 3
    else:
        n = int(args.num_tries)
        if n > 5:
            n = 5
    temperature = args.temperature
    response = openai.Completion.create(
        model="code-davinci-002",
        prompt=prompting.prompt(args),
        temperature=temperature,
        n=n,
        max_tokens=300,
        stop=["FINISH"],
        logprobs=5,
    )
    # print(response["choices"][0]["text"])
    choices = []
    for i in range(0, n):
        output = response["choices"][i]["text"]
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)


def text_davinci_003(args):
    if args.keyfile != "":
        keyfile = args.keyfile
    else:
        keyfile = os.path.join(args.keydir, "oai_key.txt")
    key = open(keyfile).readline().strip("\n")
    if key == "":
        raise Exception("No key provided.")
    openai.api_key = key
    if args.num_tries == "":
        n = 3
    else:
        n = int(args.num_tries)
        if n > 5:
            n = 5
    temperature = args.temperature
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompting.prompt(args),
        temperature=temperature,
        n=n,
        max_tokens=300,
        stop=["FINISH"],
        logprobs=5,
    )
    # print(response["choices"][0]["text"])
    choices = []
    for i in range(0, n):
        output = response["choices"][i]["text"]
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)


def code_davinci_edit_001(args):
    if args.keyfile != "":
        keyfile = args.keyfile
    else:
        keyfile = os.path.join(args.keydir, "oai_key.txt")
    key = open(keyfile).readline().strip("\n")
    if key == "":
        raise Exception("No key provided.")
    openai.api_key = key
    if args.num_tries == "":
        n = 3
    else:
        n = int(args.num_tries)
        if n > 5:
            n = 5

    temperature = args.temperature
    prompt = prompting.prompt(args) + " REPLACE"

    response = openai.Edit.create(
        model="code-davinci-edit-001",
        input=prompt,
        instruction="replace REPLACE with the explanation, an explanation dictionary and the final translation",
        temperature=temperature,
        top_p=1,
        n=n,
    )
    # print(response["choices"][0]["text"])
    choices = []
    for i in range(0, n):
        output = response["choices"][i]["text"][len(prompt) - 8 :].split("FINISH")[0]
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)


def text_bison_001(args):
    if args.keyfile != "":
        keyfile = args.keyfile
    else:
        keyfile = os.path.join(args.keydir, "google_project_id.txt")
    key = open(keyfile).readline().strip("\n")
    if key == "":
        raise Exception("No key provided.")
    vertexai.init(project=key)
    model = TextGenerationModel.from_pretrained("text-bison@001")
    n = args.num_tries

    def query():
        return model.predict(
            prompting.prompt(args), temperature=args.temperature, max_output_tokens=300
        )

    choices = []
    for i in range(0, n):
        repsonse = query()
        output = repsonse.text.split("FINISH")[0]
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)


def code_bison_001(args):
    if args.keyfile != "":
        keyfile = args.keyfile
    else:
        keyfile = os.path.join(args.keydir, "google_project_id.txt")
    key = open(keyfile).readline().strip("\n")
    if key == "":
        raise Exception("No key provided.")
    vertexai.init(project=key)
    model = CodeGenerationModel.from_pretrained("code-bison@001")
    n = args.num_tries

    def query():
        return model.predict(
            prefix=prompting.prompt(args),
            temperature=args.temperature,
            max_output_tokens=300,
        )

    choices = []
    for i in range(0, n):
        repsonse = query()
        output = repsonse.text.split("FINISH")[0]
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)


def bloom(args):
    n = args.num_tries
    input_prompt = prompting.prompt(args)
    API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
    if args.keyfile != "":
        keyfile = args.keyfile
    else:
        keyfile = os.path.join(args.keydir, "hf_key.txt")
    key = open(keyfile).readline().strip("\n")
    if key == "":
        raise Exception("No key provided.")
    headers = {"Authorization": "Bearer " + key}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    choices = []
    for i in range(0, n):
        raw_output = query(
            {
                "inputs": input_prompt,
                "options": {"use_cache": False, "wait_for_model": True},
                "parameters": {
                    "return_full_text": False,
                    "do_sample": False,
                    "max_new_tokens": 300,
                    "temperature": args.temperature,
                },
            }
        )
        # shots_count = input_prompt.count("FINISH")
        output = raw_output[0]["generated_text"].split("FINISH")[0]
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)


def bloomz(args):
    n = args.num_tries
    input_prompt = prompting.prompt(args)
    API_URL = "https://api-inference.huggingface.co/models/bigscience/bloomz"
    if args.keyfile != "":
        keyfile = args.keyfile
    else:
        keyfile = os.path.join(args.keydir, "hf_key.txt")
    key = open(keyfile).readline().strip("\n")
    if key == "":
        raise Exception("No key provided.")
    headers = {"Authorization": "Bearer " + key}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    choices = []
    for i in range(0, n):
        raw_output = query(
            {
                "inputs": input_prompt,
                "options": {"use_cache": False, "wait_for_model": True},
                "parameters": {
                    "return_full_text": False,
                    "do_sample": False,
                    "max_new_tokens": 300,
                    "temperature": args.temperature,
                },
            }
        )
        print("RAW OUTPUT")
        print(raw_output)
        # shots_count = input_prompt.count("FINISH")
        output = raw_output[0]["generated_text"].split("FINISH")[0]
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)


def deepseek_model(args):
    n = 1
    # 读取API密钥
    if args.keyfile != "":
        keyfile = args.keyfile
    else:
        keyfile = os.path.join(args.keydir, "../keyfile/key.txt")
    key = ""
    with open(keyfile, 'r') as file:
        key = file.readline().strip("\n")
    key = 'sk-e407e704f6974a88bca37589eaced336'
    if key == "":
        raise Exception("No key provided.")
    
    # 确定尝试次数
    if args.num_tries == "":
        n = 3
    else:
        n = int(args.num_tries)
        if n > 5:
            n = 5
    
    temperature = args.temperature
    prompt_text = prompting.prompt(args)  # 假设 prompting 是一个已定义的模块
    
    # DeepSeek API的URL(利用阿里百炼平台创建)
    url = "https://api.deepseek.com"  
    
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key = key,
        base_url = url
    )

    completion = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {'role': 'user', 
             'content': prompt_text,
             "temperature": temperature,
             "n": n,
             'max_tokens': 300,
             "stop": ["FINISH"]}
        ]
    )

    reasoning_content = completion.choices[0].message.reasoning_content
    content = completion.choices[0].message.content
    print('reasoning_content: ', reasoning_content)
    print('content: ', content)

    choices = []
    for i in range(0, n):
        output = content
        print("OUTPUT")
        print(output)
        choices.append(output)
    return prompting.extract_subinfo(choices, args, n)

def llama_model(args):
    n = 1
    # 读取API密钥
    if args.keyfile != "":
        keyfile = args.keyfile
    else:
        keyfile = os.path.join(args.keydir, "../keyfile/key.txt")
    key = ""
    with open(keyfile, 'r') as file:
        key = file.readline().strip("\n")
    
    if key == "":
        raise Exception("No key provided.")
    
    # 确定尝试次数
    if args.num_tries == "":
        n = 3
    else:
        n = int(args.num_tries)
        if n > 5:
            n = 5
    
    prompt_text = prompting.prompt(args)  # 假设 prompting 是一个已定义的模块

    dashscope.api_key = key

    messages = [{'role': 'user', 
                 'content': prompt_text}]
    
    response = dashscope.Generation.call(
        api_key = os.getenv('DASHSCOPE_API_KEY'),
        model = 'llama3.3-70b-instruct',
        messages = messages,
        max_tokens = 1000,
        stop = ["FINISH"],
        n = n,
        temperature = args.temperature,
        result_format='message',  # set the result to be "message" format.
    )

    
    if response.status_code == HTTPStatus.OK:
        print('----------------------------------------')
        print(response["output"]["choices"][0]["message"]["content"])
        print('----------------------------------------')

    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))

    
    choices = []
    
    for i in range(0, n):
        output = response["output"]["choices"][0]["message"]["content"]
        # print("OUTPUT")
        # print(output)
        choices.append(output)
    
    return prompting.extract_subinfo(choices, args, n)

def llama3b_model(args):
    n = 1
    # 读取API密钥
    if args.keyfile != "":
        keyfile = args.keyfile
    else:
        keyfile = os.path.join(args.keydir, "../keyfile/key.txt")
    key = ""
    with open(keyfile, 'r') as file:
        key = file.readline().strip("\n")
    
    if key == "":
        raise Exception("No key provided.")
    
    # 确定尝试次数
    if args.num_tries == "":
        n = 3
    else:
        n = int(args.num_tries)
        if n > 5:
            n = 5
    
    prompt_text = prompting.prompt(args)  # 假设 prompting 是一个已定义的模块

    dashscope.api_key = key

    messages = [{'role': 'user', 
                 'content': prompt_text}]
    
    response = dashscope.Generation.call(
        api_key = os.getenv('DASHSCOPE_API_KEY'),
        model = 'llama3.2-3b-instruct',
        messages = messages,
        max_tokens = 1000,
        stop = ["FINISH"],
        n = n,
        temperature = args.temperature,
        result_format='message',  # set the result to be "message" format.
    )

    
    if response.status_code == HTTPStatus.OK:
        print('----------------------------------------')
        print(response["output"]["choices"][0]["message"]["content"])
        print('----------------------------------------')

    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))

    
    choices = []
    
    for i in range(0, n):
        output = response["output"]["choices"][0]["message"]["content"]
        # print("OUTPUT")
        # print(output)
        choices.append(output)
    
    return prompting.extract_subinfo(choices, args, n)


def qwen_model(args):
    n = 1
    # 读取API密钥
    if args.keyfile != "":
        keyfile = args.keyfile
    else:
        keyfile = os.path.join(args.keydir, "../keyfile/key.txt")
    key = ""
    with open(keyfile, 'r') as file:
        key = file.readline().strip("\n")
    
    if key == "":
        raise Exception("No key provided.")
    
    # 确定尝试次数
    if args.num_tries == "":
        n = 3
    else:
        n = int(args.num_tries)
        if n > 5:
            n = 5
    
    temperature = args.temperature
    prompt_text = prompting.prompt(args)  # 假设 prompting 是一个已定义的模块
    
    # DeepSeek API的URL(利用阿里百炼平台创建)
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  
    
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key = key,
        base_url = url
    )

    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复
    is_answering = False   # 判断是否结束思考过程并开始回复

    completion = client.chat.completions.create(
        model="qwq-32b",  
        messages=[
            {'role': 'user', 
             'content': prompt_text,
             "temperature": temperature,
             "n": n,
             'max_tokens': 300,
             "stop": ["FINISH"]}
        ],
        stream = True
    )

    print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")
    ans = ''
    for chunk in completion:
        # 如果chunk.choices为空，则打印usage
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
        else:
            delta = chunk.choices[0].delta
            # 打印思考过程
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                #print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            else:
                # 开始
                if delta.content != "" and is_answering is False:
                    print("\n" + "=" * 20 + "answering" + "=" * 20 + "\n")
                    is_answering = True
                # 打印回复过程
                print(delta.content, end='', flush=True)
                answer_content += delta.content

    choices = []
    for i in range(0, n):
        output = answer_content
        print("OUTPUT")
        print(output)
        choices.append(output)

    return prompting.extract_subinfo(choices, args, n)

def qwen7b_model(args):
    n = 1
    # 读取API密钥
    if args.keyfile != "":
        keyfile = args.keyfile
    else:
        keyfile = os.path.join(args.keydir, "../keyfile/key.txt")
    key = ""
    with open(keyfile, 'r') as file:
        key = file.readline().strip("\n")
    
    if key == "":
        raise Exception("No key provided.")
    
    # 确定尝试次数
    if args.num_tries == "":
        n = 3
    else:
        n = int(args.num_tries)
        if n > 5:
            n = 5
    
    temperature = args.temperature
    prompt_text = prompting.prompt(args)  # 假设 prompting 是一个已定义的模块
    
    # DeepSeek API的URL(利用阿里百炼平台创建)
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  
    
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key = key,
        base_url = url
    )

    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复
    is_answering = False   # 判断是否结束思考过程并开始回复

    completion = client.chat.completions.create(
        model="qwen-plus",  
        messages=[
            {'role': 'user', 
             'content': prompt_text,
             "temperature": temperature,
             "n": n,
             'max_tokens': 300,
             "stop": ["FINISH"]}
        ],
        stream = True
    )

    print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")
    ans = ''
    for chunk in completion:
        # 如果chunk.choices为空，则打印usage
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
        else:
            delta = chunk.choices[0].delta
            # 打印思考过程
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                #print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            else:
                # 开始
                if delta.content != "" and is_answering is False:
                    print("\n" + "=" * 20 + "answering" + "=" * 20 + "\n")
                    is_answering = True
                # 打印回复过程
                print(delta.content, end='', flush=True)
                answer_content += delta.content

    choices = []
    for i in range(0, n):
        output = answer_content
        print("OUTPUT")
        print(output)
        choices.append(output)

    return prompting.extract_subinfo(choices, args, n)


def FCQs_llama3b_model(args):
    n = 1
    # 读取API密钥
    if args.keyfile != "":
        keyfile = args.keyfile
    else:
        keyfile = os.path.join(args.keydir, "../keyfile/key.txt")
    key = ""
    with open(keyfile, 'r') as file:
        key = file.readline().strip("\n")
    
    if key == "":
        raise Exception("No key provided.")
    
    # 确定尝试次数
    if args.num_tries == "":
        n = 3
    else:
        n = int(args.num_tries)
        if n > 5:
            n = 5
    
    prompt_text = prompting.FCQs_prompt(args)  # 假设 prompting 是一个已定义的模块

    dashscope.api_key = key

    messages = [{'role': 'user', 
                 'content': prompt_text}]
    
    response = dashscope.Generation.call(
        api_key = os.getenv('DASHSCOPE_API_KEY'),
        model = 'llama3.2-3b-instruct',
        messages = messages,
        max_tokens = 1000,
        stop = ["FINISH"],
        n = n,
        temperature = args.temperature,
        result_format='message',  # set the result to be "message" format.
    )

    '''
    if response.status_code == HTTPStatus.OK:
        print('------------------ddddd----------------------')
        print(response["output"]["choices"][0]["message"]["content"])
        print('------------------eeeee---------------------')

    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
    '''
    output = response["output"]["choices"][0]["message"]["content"]
    
    return prompting.FCQs_extract_subinfo(output, args)

def FCQs_qwen7b_model(args):
    n = 1
    # 读取API密钥
    if args.keyfile != "":
        keyfile = args.keyfile
    else:
        keyfile = os.path.join(args.keydir, "../keyfile/key.txt")
    key = ""
    with open(keyfile, 'r') as file:
        key = file.readline().strip("\n")
    
    if key == "":
        raise Exception("No key provided.")
    
    # 确定尝试次数
    if args.num_tries == "":
        n = 3
    else:
        n = int(args.num_tries)
        if n > 5:
            n = 5
    
    temperature = args.temperature
    prompt_text = prompting.FCQs_prompt(args)  # 假设 prompting 是一个已定义的模块
    
    # DeepSeek API的URL(利用阿里百炼平台创建)
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1"  
    
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key = key,
        base_url = url
    )

    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复
    is_answering = False   # 判断是否结束思考过程并开始回复

    completion = client.chat.completions.create(
        model="qwen-plus",  
        messages=[
            {'role': 'user', 
             'content': prompt_text,
             "temperature": temperature,
             "n": n,
             'max_tokens': 300,
             "stop": ["FINISH"]}
        ],
        stream = True
    )

    print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")
    ans = ''
    for chunk in completion:
        # 如果chunk.choices为空，则打印usage
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
        else:
            delta = chunk.choices[0].delta
            # 打印思考过程
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                #print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            else:
                # 开始
                if delta.content != "" and is_answering is False:
                    print("\n" + "=" * 20 + "answering" + "=" * 20 + "\n")
                    is_answering = True
                # 打印回复过程
                print(delta.content, end='', flush=True)
                answer_content += delta.content

    return prompting.FCQs_extract_subinfo(answer_content, args)