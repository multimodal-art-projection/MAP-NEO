import asyncio
from multiprocessing import Pool
import os
import time
import concurrent.futures
from openai import APIError, AsyncOpenAI

chinese_prompt = """请扮演一个AI校对员，我需要你的专业技能来帮助我校对一段文本。这段文本是我通过OCR技术从一份原始文档中提取出来的，我怀疑在字符识别的过程中可能发生了一些错误。具体来说，可能存在拼写错误、语法错误、标点用错或者格式排列问题。请特别注意生成的内容中有很多识别错误的空格与换行符。请将段落整理成正确的语义通顺的格式。下面是我提供的文本内容，请你帮我仔细检查并校对，请直接输出修订后文本，并不要包含其他内容。"""
english_prompt = """from an original document using OCR technology, there may be errors in character recognition, potentially including spelling mistakes, grammatical errors, incorrect punctuation, or formatting issues. Pay special attention to misplaced spaces and line breaks that often occur in OCR-generated content. I need you to reorganize the paragraph into a properly formatted and semantically coherent form. Here's the text I've provided. Kindly check and correct it meticulously. Please output only the revised text without including any additional content i.e. any comments from you. The output format should be a well organized markdown content. Do not change the language, i.e. Do not change chinese content to English. Some contents are mixed language i.e. Chinese main content with English symbols. Also, do not change the original language. 注意不要产生任何额外的批注！
Here's one of the texts that needs to be processed:"""

pdf_convert_prompt = """
# 角色
你是一位专业的PDF转写与校对专家，专注于将PDF文档精确转换为Markdown格式，并具备精湛的文本纠错与排版技能。

## 技能
### 技能1：PDF转Markdown转换
- 使用高级技术精确提取PDF内容，包括文本、图片引用标注等，转换为Markdown格式。
- 保持原文档的结构和格式，如标题层级、列表、代码块等，在Markdown中的正确表示。

### 技能2：文本纠错与格式修正
- 识别并修正转换过程中可能产生的拼写错误、语法错误和标点符号误用。
- 特别留意并修正因识别错误导致的额外空格、换行问题，确保文本流畅度和阅读体验。
- 重组段落，确保语义连贯，符合常规的阅读逻辑和Markdown格式规范。

### 技能3：细节审查与优化
- 细致检查文档中的每一个细节，包括但不限于链接的有效性、图片描述的准确性等。
- 确保Markdown输出适用于各种Markdown解析器，兼容性广泛。

## 限制
- 专注文本内容的转换与校对，不涉及PDF本身的编辑或修改。
- 转写过程中对于高度格式化的表格或复杂图表，可能需简化处理以适应Markdown格式。
- 服务范围不包括原文档内容的实质编辑或增加额外注释，除非明确指出是由于转换错误所致。

## 任务执行
- 直接输出经过仔细检查与修订后的Markdown文本，排除任何非内容相关的备注或说明。
- 确保最终输出文档既忠于原意，又在Markdown格式下呈现最佳阅读体验。
"""

class AsyncLLMQueryHandler:
    def __init__(self, api_key=os.environ.get('api_key')):
        self.api_key = api_key
        self.aclient = AsyncOpenAI(api_key=self.api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    async def async_query_openai(self, query, prompt=english_prompt, **kwargs):
        try:
            completion = await self.aclient.chat.completions.create(
                model="qwen-long",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ],
                stream=False,
                **kwargs  # 使用 **kwargs 来允许额外的参数传递给 completions.create
            )
            # 确保choices非空
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content
            else:
                raise ValueError("API 返回的 choices 为空或不包含 message")
        except APIError as e:
            # 处理 API 调用错误
            print(f"API 调用失败: {e}")
            # print query
            print(f"query: {query}\n")
            return None
        except Exception as e:
            # 处理其他意外异常
            print(f"处理查询时发生错误: {e}")
            return None

    async def async_process_queries(self, queries, **kwargs):
        tasks = [
            self.async_query_openai(query, **kwargs)  # 将kwargs传递给每个查询
            for query in queries
        ]
        results = await asyncio.gather(*tasks)
        return results
async def main():

    handler = AsyncLLMQueryHandler(api_key=os.environ.get('api_key'))
    queries = ["介绍三个北京必去的旅游景点。",
               "介绍三个成都最有名的美食。",
               "介绍三首泰勒斯威夫特好听的歌曲",
               "蓝牙耳机坏了需要看医院的哪个科室？",
               "介绍三个中国好听的歌",
               "左手一只鸭，右手一只鸡。交换两次后左右手里各是什么？",
               "为什么鲁智深不能倒拔垂杨柳而林黛玉却可以？",]
    start_time = time.time()  # 开始计时

    results = await handler.async_process_queries(queries, prompt=chinese_prompt)
    end_time = time.time()  # 结束计时
    for result in results:
        print(result)
        print("-" * 50)
    print(f"Total time: {end_time - start_time:.2f} seconds")

def thread_pool_executor_for_async(async_func, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(1) as executor:
        future = executor.submit(lambda: asyncio.run(async_func(*args, **kwargs)))
        return future.result()
def process_predict(queries):
    handler = AsyncLLMQueryHandler()
    results = thread_pool_executor_for_async(handler.async_process_queries, queries)
    print(results)
    return results  # 或者任何你需要的处理结果

if __name__ == "__main__":

    queries = ["介绍三个北京必去的旅游景点。",
            "介绍三个成都最有名的美食。",
            "介绍三首泰勒斯威夫特好听的歌曲",
            "蓝牙耳机坏了需要看医院的哪个科室？",
            "介绍三个中国好听的歌",
            "左手一只鸭，右手一只鸡。交换两次后左右手里各是什么？",
            "为什么鲁智深不能倒拔垂杨柳而林黛玉却可以？",]
    # asyncio.run(main())
    # handler = AsyncLLMQueryHandler()
    # results = thread_pool_executor_for_async(handler.async_process_queries, queries)
    # print(results)
    # print('----------------------')
    with Pool(4)  as pool:
        results = pool.map(process_predict, [queries for _ in range(10)])
