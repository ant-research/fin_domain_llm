from weaverbird.models import load_model_and_tokenizer
from weaverbird.utils import parse_configs


def main():
    model_config_dict = {'model_name_or_dir': 'chatglm2-6b'}

    configs = parse_configs(model_config_dict)

    load_model_and_tokenizer(configs['model_config'])

    return


if __name__ == '__main__':
    main()