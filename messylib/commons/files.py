import pathlib, os
from natsort import natsorted

# --------------------------------------------------------------------


def get_filepaths_bytype(
    target_path: str,
    filetype: str,
    exclude_hidden: bool = True,
    make_string: bool = True,
    verbose: bool = True,
) -> list:
    """
    Get paths from a directory with spesific file type.

    """
    target_path = pathlib.Path(target_path)
    paths = list(target_path.glob(f"*{filetype}"))
    if exclude_hidden:
        paths = [p for p in paths if not os.path.basename(p).startswith(".")]
    if make_string:
        paths = [str(p) for p in paths]
    paths = natsorted(paths)
    if verbose == True:
        print(f"Total paths: {len(paths)}")
    return paths


# --------------------------------------------------------------------


def create_filedict(pathlist: list, filetype: str) -> dict:
    """Collect files into filesdict and returns filesdict."""
    filesdict = {}
    for path in pathlist:
        fname = os.path.basename(path)
        fid = fname.split(filetype)[0]
        filesdict[fid] = path
    return filesdict
