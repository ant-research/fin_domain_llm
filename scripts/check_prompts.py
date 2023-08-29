from weaverbird.prompts import ChatPromptTemplate


if __name__ == '__main__':
    prompt = ChatPromptTemplate(lang='en').from_template(context='hello', date='20200930', question='what is nasdaq close price')
    print(prompt)