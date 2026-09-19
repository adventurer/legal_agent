import os
from openai import OpenAI

print("环境变量 OPENAI_BASE_URL:", os.environ.get("OPENAI_BASE_URL"))

# 显式传入标准 URL
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="none",
)

try:
    resp = client.models.list()
    print("连接成功！获取到的模型列表:")
    for m in resp.data:
        print(" -", m.id)
except Exception as e:
    print("报错详情:", type(e), e)
    import traceback
    traceback.print_exc()