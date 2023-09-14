import os
from typing import List, Optional

from langchain.document_loaders import TextLoader, UnstructuredMarkdownLoader, UnstructuredFileLoader
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import TextSplitter
from tqdm import tqdm

from weaverbird.cn_text_splitter import ChineseTextSplitter
from weaverbird.utils import logger


def tree(filepath, ignore_dir_names=None, ignore_file_names=None):
    """
    Return two list, the first one is the dirs of all files under filepath, the second one is the file names of all
    corresponding files.
    borrowed from https://github.com/chatchat-space/Langchain-Chatchat/

    """
    if ignore_dir_names is None:
        ignore_dir_names = []
    if ignore_file_names is None:
        ignore_file_names = []
    ret_list = []
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            print("Directory not existed")
            return None, None
        elif os.path.isfile(filepath) and os.path.basename(filepath) not in ignore_file_names:
            return [filepath], [os.path.basename(filepath)]
        elif os.path.isdir(filepath) and os.path.basename(filepath) not in ignore_dir_names:
            for file in os.listdir(filepath):
                fullfilepath = os.path.join(filepath, file)
                if os.path.isfile(fullfilepath) and os.path.basename(fullfilepath) not in ignore_file_names:
                    ret_list.append(fullfilepath)
                if os.path.isdir(fullfilepath) and os.path.basename(fullfilepath) not in ignore_dir_names:
                    ret_list.extend(tree(fullfilepath, ignore_dir_names, ignore_file_names)[0])
    return ret_list, [os.path.basename(p) for p in ret_list]


def load_file(file_dir, text_splitter):
    if file_dir.lower().endswith(".md"):
        loader = UnstructuredMarkdownLoader(file_dir)
    elif file_dir.lower().endswith(".txt"):
        loader = TextLoader(file_dir, autodetect_encoding=True)
    else:
        loader = UnstructuredFileLoader(file_dir, mode="elements")

    docs = loader.load_and_split(text_splitter=text_splitter)

    return docs


class LocalKnowledgeBaseLoader(BaseLoader):
    def __init__(self,
                 file_dir: str or List[str],
                 text_splitter: Optional[TextSplitter] = ChineseTextSplitter
                 ):
        self.file_dir = file_dir
        self.text_splitter = text_splitter

    def _load_from_single_dir(self, file_dir):
        docs = []
        loaded_files = []
        failed_files = []
        if not os.path.exists(file_dir):
            logger.info("Directory not existed")
            return None
        elif os.path.isfile(file_dir):
            file = os.path.split(file_dir)[-1]
            try:
                docs = load_file(file_dir, self.text_splitter)
                logger.info(f"{file} loaded")
                loaded_files.append(file_dir)
            except Exception as e:
                logger.error(e)
                logger.info(f"{file} failed to load")
                failed_files.append(file)
        elif os.path.isdir(file_dir):
            docs = []
            for single_file_dir, file in tqdm(zip(*tree(file_dir)), desc="loading files"):
                try:
                    docs += load_file(single_file_dir, self.text_splitter)
                    loaded_files.append(single_file_dir)
                except Exception as e:
                    logger.error(e)
                    failed_files.append(single_file_dir)

        return docs, loaded_files, failed_files

    def _load_from_multiple_dir(self, file_dir):
        docs = []
        loaded_files = []
        failed_files = []

        for file in file_dir:
            docs_, loaded_files_, failed_files_ = self._load_from_single_dir(file)
            docs.extend(docs_)
            loaded_files.extend(loaded_files_)
            failed_files.extend(failed_files_)
        return docs, loaded_files, failed_files

    def load(self):
        if isinstance(self.file_dir, str):
            docs, loaded_files, failed_files = self._load_from_single_dir(self.file_dir)
        else:
            docs, loaded_files, failed_files = self._load_from_multiple_dir(self.file_dir)

        return docs
