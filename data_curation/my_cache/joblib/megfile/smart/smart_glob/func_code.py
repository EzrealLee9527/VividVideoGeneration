# first line: 656
def smart_glob(
        pathname: PathLike, recursive: bool = True,
        missing_ok: bool = True) -> List[str]:
    '''
    Given pathname may contain shell wildcard characters, return path list in ascending alphabetical order, in which path matches glob pattern

    :param pathname: A path pattern may contain shell wildcard characters
    :param recursive: If False, this function will not glob recursively
    :param missing_ok: If False and target path doesn't match any file, raise FileNotFoundError
    '''
    # Split pathname, group by protocol, call glob respectively
    # SmartPath(pathname).glob(recursive, missing_ok)
    result = []
    group_glob_list = _group_glob(pathname)
    for glob_path in group_glob_list:
        for path_obj in SmartPath(glob_path).glob(pattern='',
                                                  recursive=recursive,
                                                  missing_ok=missing_ok):
            result.append(path_obj.path)
    return result
