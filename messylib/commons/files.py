import pathlib, os
from natsort import natsorted

# --------------------------------------------------------------------


def get_paths_filetype(
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
