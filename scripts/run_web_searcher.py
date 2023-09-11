from weaverbird.utils import parse_configs
from weaverbird.retrievers import WebSearcher

def main():
    search_config = {'model_name_or_path': None,
                     'serp_api_token': 'xxx'}

    configs = parse_configs(search_config)

    search_config = configs['websearch_config']

    web_searcher_cls = WebSearcher.build_from_config(search_config)

    results = web_searcher_cls.get_relevant_documents('what does Elon Musk think of BYD')

    print(results)

    return


if __name__ == '__main__':
    main()