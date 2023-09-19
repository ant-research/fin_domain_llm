from langchain import LLMChain

from weaverbird.chains.chat_retro.prompt import CHAT_RETRO_EN_PROMPT
from weaverbird.models import LLMChatModel
from weaverbird.utils import parse_configs


def main():
    model_config_dict = {'model_name_or_path': 'chatglm2-6b'}

    configs = parse_configs(model_config_dict)

    chat_model = LLMChatModel(configs['model_config'], generation_config=configs['generation_config'])

    chat_prompt = CHAT_RETRO_EN_PROMPT

    chain = LLMChain(prompt=chat_prompt, llm=chat_model, verbose=True)

    print(chain({'context': 'hello', 'date': '20200930', 'question': 'what is nasdaq close price'}))

    return


if __name__ == '__main__':
    main()
