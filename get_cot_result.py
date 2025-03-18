import json
from collections import defaultdict
from openai import OpenAI



def read_and_process_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        case_data = json.load(f)

    
    responses = []
    for i, case in enumerate(case_data):
        print(f"正在处理案例{i+1}")
        zp_prompt = "假设您擅长识别诈骗案例，请根据以下情节分析并判断该案例是否为诈骗行为。如果是，请输出1；否则，请输出0。只需输出1或0。"
        print(case.keys())
        user_content = f"{zp_prompt}\n{case['content']}"
        print(user_content)
        print("="*100)


        # 初始化客户端，需替换为实际部署的API地址和密钥
        client = OpenAI(
            api_key="sk-6yA4XIWxUq6oIwMtBHKaccY7EKFgd7h6Q14iIaJxdAgsuwlk",  
            base_url="https://tbnx.plus7.plus/v1"  # 替换为你的API地址
        )

        # 构造对话请求
        response = client.chat.completions.create(
            model="deepseek-reasoner",  # R1模型标识符[[1,2,15]]
            messages=[
                {"role": "system", "content": "你是一个专业的助手"},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7,
            stream=False  # 如需流式响应可设为True
        )

        output = response.choices[0].message.content
        print(output)
        # 输出响应结果
        responses.append({
            "content": case['content'],
            "label": case['lable'],
            "output": output
        })


    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)



def process_result_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(data[0])
    data = [
        {'content': item['content'], 
         'label': item['label'], 
         'thinking_process': item['output'],
         'model_output':  item['output'].split("</think>\n\n")[1],
        } for item in data
    ]
    
    with open(file="result1.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# 运行脚本
if __name__ == "__main__":
    # read_and_process_json(file_path='/home/ldn/baidu/reft-pytorch-codes/zp/data.json')
    process_result_json("result.json")