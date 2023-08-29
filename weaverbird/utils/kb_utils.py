import os


def get_kbs_list(kb_root_dir):
    lst_default = ["None"]
    if not os.path.exists(kb_root_dir):
        return lst_default
    lst = os.listdir(kb_root_dir)
    if not lst:
        return lst_default
    lst.sort()
    return lst_default + lst