from weaverbird.chains.chat_retro.prompt import CHAT_RETRO_EN_PROMPT

if __name__ == '__main__':
    prompt = CHAT_RETRO_EN_PROMPT.format(context='hello', date='20200930', question='what is nasdaq close price')
    print(prompt)
