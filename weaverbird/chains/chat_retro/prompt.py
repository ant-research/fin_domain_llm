from langchain import PromptTemplate

_EN_TEMPLATE = """Through an search, you have obtained some information. Each line represents a piece of information, and each piece is independent. It includes the posting time, title, and a snippet of the information. The closer the posting time is to the present, the more important the information. The information is not complete, and the ellipsis (...) indicates omitted sections. Here is the search result:
    {context}

    The current date is {date}. You need to answer the user's questions based on the information provided above. If there are multiple questions in a query, please answer all of them. If the user's question includes keywords like 'recent' or 'latest' to indicate a recent time frame, pay attention to the correspondence between the current date and the date of the information. You MUST respond in the same language as the question! The question is: {question}"""

_CN_TEMPLATE = """通过搜索你得到了一些信息，每一行是一条信息，每条信息是独立的，其中包含了这条信息的发布时间，标题和片段，发布时间离现在越近的信息越重要，信息并不是完整的，句子中的"..."表示省略部分，以下为搜索到的信息：
    {context}

    当前日期为{date}。你需要根据上面这些信息来回答用户的问题。如果提问中有多个问题，请一并回答。如果用户的问题中提到了类似“最近”或“最新”这样表示近期的关键词，需要注意当前日期和信息的日期对应关系。要求回答完整，答案必须使用和问题同样的语种! 问题是：{question}"""

CHAT_RETRO_EN_PROMPT = PromptTemplate(input_variables=['context', 'date', 'question'],
                                      template=_EN_TEMPLATE)

CHAT_RETRO_CN_PROMPT = PromptTemplate(input_variables=['context', 'date', 'question'],
                                      template=_CN_TEMPLATE)
