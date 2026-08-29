# from openai import OpenAI
# import httpx
# import json
# import urllib3
# import base64

# # 创建定制的会话
# transport = httpx.HTTPTransport(proxy=None, verify=False)
# http_client = httpx.Client(transport=transport)

# extra_headers = {
#     "X-HW-ID": "com.huawei.ipd.coretool.coreai",  # 必填
#     "X-HW-APPKEY": "WxhsDOVQJGVYpkDfQ7C2HA=="  # 必填
# }

# extra_body = {
#     "appId": "com.huawei.ipd.coretool.coreai",  # 必填，同X-HW-ID
#     "scene": "test",  # 必填
#     "operator": "h00965148",  # 必填 工号
#     "temperature": 0.2,
#     "chat_template_kwargs": {"enable_thinking": False}
# }

# client = OpenAI(
#     base_url="https://apigw-cn-south02.huawei.com/api/v1",
#     api_key="com.huawei.ipd.coretool.coreai",  # 必填，同X-HW-ID
#     http_client=http_client
# )

# #  base 64 编码格式
# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")
# def encode_image_from_url(url, client_transport):
#     # 改名避免与外部 OpenAI 的 client 混淆
#     with httpx.Client(transport=client_transport) as http_stream_client:
#         response = http_stream_client.get(url)
#         # 修正：这是方法调用，用来检查 4xx 或 5xx 状态码
#         response.raise_for_status()  
#         return base64.b64encode(response.content).decode("utf-8")

# path = "https://fuyao-data-server.rnd.huawei.com/manual_upload/img_test/MAE-VNF_LCM_%E7%BD%91%E5%85%83%E5%B8%B8%E7%94%A8%E6%93%8D%E4%BD%9C%E5%AE%9A%E4%BD%8D%E5%AE%9A%E7%95%8C%E6%A1%88%E4%BE%8B%E9%9B%86docx_rId26.png"
# base64_image = encode_image_from_url(path, transport)

# try:
#     prompt = '图片里有什么'

#     completion = client.chat.completions.create(
#         model="6d2c5ff6-615d-45a8-9703-2f591d6c2437",
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image_url",
#                         "image_url": {"url": f"data:image/png;base64,{base64_image}"}
#                     },
#                     {"type": "text", "text": prompt}
#                 ]
#             }
#         ],
#         extra_body=extra_body,
#         extra_headers=extra_headers,
#         stream=True
#     )
#     res = ""
#     for chunk in completion:
#         if chunk.choices:
#             res += chunk.choices[0].delta.content
#             print(chunk.choices[0].delta.content, end="")
#         # print(res)
# except Exception as err:
#     print(f"错误发生: {err}")


from openai import OpenAI
import httpx
import json
import urllib3
import base64

# 创建定制的会话
transport = httpx.HTTPTransport(proxy=None, verify=False)
http_client = httpx.Client(transport=transport)

extra_headers = {
    "X-HW-ID": "com.huawei.ipd.coretool.coreai",  # 必填
    "X-HW-APPKEY": "WxhsDOVQJGVYpkDfQ7C2HA=="  # 必填
}

extra_body = {
    "appId": "com.huawei.ipd.coretool.coreai",  # 必填，同X-HW-ID
    "scene": "test",  # 必填
    "operator": "h00965148",  # 必填 工号
    "temperature": 0.2,
    "chat_template_kwargs": {"enable_thinking": False}
}

client = OpenAI(
    base_url="https://apigw-cn-south02.huawei.com/api/v1",
    api_key="com.huawei.ipd.coretool.coreai",  # 必填，同X-HW-ID
    http_client=http_client
)

#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
def encode_image_from_url(url, client_transport):
    # 改名避免与外部 OpenAI 的 client 混淆
    with httpx.Client(transport=client_transport) as http_stream_client:
        response = http_stream_client.get(url)
        # 修正：这是方法调用，用来检查 4xx 或 5xx 状态码
        response.raise_for_status()  
        return base64.b64encode(response.content).decode("utf-8")

path = "https://fuyao-data-server.rnd.huawei.com/manual_upload/img_test/MAE-VNF_LCM_%E7%BD%91%E5%85%83%E5%B8%B8%E7%94%A8%E6%93%8D%E4%BD%9C%E5%AE%9A%E4%BD%8D%E5%AE%9A%E7%95%8C%E6%A1%88%E4%BE%8B%E9%9B%86docx_rId26.png"
base64_image = encode_image_from_url(path, transport)

try:
    prompt = '你好'

    completion = client.chat.completions.create(
        model="fa6c020a-06e3-4a4f-8840-2951e5ef934d",
        messages=[
            {
                "role": "user",
                "content": [
                    # {
                    #     "type": "image_url",
                    #     "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    # },
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        extra_body=extra_body,
        extra_headers=extra_headers,
        stream=True
    )
    res = ""
    for chunk in completion:
        if chunk.choices:
            res += chunk.choices[0].delta.content
            print(chunk.choices[0].delta.content, end="")
        # print(res)
except Exception as err:
    print(f"错误发生: {err}")