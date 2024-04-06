from weaverbird.retrievers import BaseRetro
from weaverbird.utils import parse_configs


def main():
    search_config = {'model_name_or_path': None,
                     'serp_api_token': 'xxx'}

    configs = parse_configs(search_config)

    search_config = configs['retro_config']

    web_searcher_cls = BaseRetro.build_from_config(search_config)

    results = web_searcher_cls.get_relevant_documents('what does Elon Musk think of BYD')

    print(results)

    return


if __name__ == '__main__':
    main()
