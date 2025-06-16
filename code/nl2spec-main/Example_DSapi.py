import os
from openai import OpenAI
from http import HTTPStatus
import dashscope

def deepseek_api():
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-fe2c9005873f468b9d67a1577d74d439",  # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    completion = client.chat.completions.create(
        model="deepseek-r1",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
        messages=[
            {'role': 'user', 'content': '9.9和9.11谁大'}
        ]
    )

    # 通过reasoning_content字段打印思考过程
    print("思考过程：")
    print(completion.choices[0].message.reasoning_content)

    # 通过content字段打印最终答案
    print("最终答案：")
    print(completion.choices[0].message.content)

def llama_api():
    dashscope.api_key = 'sk-fe2c9005873f468b9d67a1577d74d439'

    messages = [{'role': 'user', 'content': '介绍一下自己'}]
    response = dashscope.Generation.call(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model='llama3.3-70b-instruct',
        messages=messages,
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))

if __name__ == '__main__':
    llama_api()
