from weaverbird.models import load_model_and_tokenizer
from weaverbird.utils import parse_configs


def main():
    model_config_dict = {'model_name_or_dir': '/mnt/nas/users/antflux/LM/chatglm2-6b'}

    model_config, _, _ = parse_configs(model_config_dict)

    load_model_and_tokenizer(model_config)
    # load_model_and_tokenizer(parse_configs(model_config))

    return


if __name__ == '__main__':
    main()