def mount_drive(force_remount=False):
    """
    Mount Google Drive into the Colab runtime filesystem.

    Args:
        force_remount (bool): If True, unmounts and remounts Drive even if
                              already mounted. Useful when Drive state is stale.

    Returns:
        Path: A pathlib.Path object pointing to the root of 'My Drive',
              i.e. /content/drive/My Drive
    """
    from google.colab import drive
    from pathlib import Path

    drive.mount('/content/drive', force_remount=force_remount)
    return Path.cwd().joinpath('drive/My Drive')

def unmount_drive():
    """
    Safely flush pending writes and unmount Google Drive from the runtime.

    Call this before ending a session to ensure all buffered data is
    written back to Drive and the mount is cleanly released.
    """
    from google.colab import drive
    drive.flush_and_unmount()


def download_file(src, filename, **kwargs):
    """
    Serialize and download a data object from the Colab runtime to the
    user's local machine.

    Supported formats are inferred from the file extension:
        - Images  (.jpeg, .jpg, .png, .pdf) : saved via matplotlib's savefig
        - CSV     (.csv)                    : saved via DataFrame.to_csv
        - Excel   (.xlsx)                   : saved via DataFrame.to_excel
        - Other                             : serialized with pickle

    Args:
        src      : The object to export. Expected types per extension:
                     - matplotlib Figure  for image/PDF outputs
                     - pandas DataFrame   for CSV/Excel outputs
                     - any picklable obj  for all other extensions
        filename (str): Destination filename including extension
                        (e.g. 'results/fig.png', 'data.csv').
                        The extension controls the serialization path.
        **kwargs : Extra keyword arguments forwarded to matplotlib's
                   savefig (only applied for image/PDF exports).

    Side Effects:
        Writes a file to the Colab runtime filesystem at `filename`,
        then triggers a browser download of that file.
    """
    import pickle
    from google.colab import files

    ext = filename.split(".")[-1]

    # --- Image / PDF export ---
    if ext in ["jpeg", "jpg", "png", "pdf"]:
        fig = src
        fig.savefig(filename, bbox_inches="tight", **kwargs)

    # --- CSV export ---
    elif ext == "csv":
        df = src
        df.to_csv(filename)

    # --- Excel export ---
    elif ext == "xlsx":
        df = src
        df.to_excel(filename)

    # --- Fallback: binary pickle ---
    else:
        with open(filename, "wb") as f:
            pickle.dump(src, f)

    files.download(filename)


def clear_temp_storage(ignore_folders=[]):
    """
    Delete all user-created files and directories under /content,
    skipping protected system folders and any caller-specified exceptions.

    Protected folders (always preserved):
        .config, drive, .ipynb_checkpoints, sample_data

    Args:
        ignore_folders (list[str]): Additional folder names to preserve.
                                    Matched against the top-level name only
                                    (not the full path).

    Raises:
        ValueError: If an entry under /content is neither a file nor a
                    directory (e.g. a dangling symlink).

    Notes:
        - Directory removal uses ignore_errors=True, so permission issues
          on individual files will not abort the entire cleanup.
        - Only the immediate children of /content are iterated; the function
          recurses implicitly via shutil.rmtree for directories.
    """
    import shutil
    from pathlib import Path

    # Folders that must never be removed
    protected = ['.config', 'drive', '.ipynb_checkpoints', 'sample_data']

    for p in Path('/content').glob('*'):
        if p.is_dir():
            if p.name in protected + ignore_folders:
                continue  # Skip protected and user-excluded directories
            shutil.rmtree(p, ignore_errors=True)
        elif p.is_file():
            p.unlink()
        else:
            raise ValueError(f'It must be a file or directory: {p}')


def download_temp_storage(ignore_folders=[], target_folder=''):
    """
    Download files from the Colab runtime's /content directory to the
    user's local machine.

    Two operating modes depending on `target_folder`:

    Targeted mode (target_folder provided):
        Recursively downloads every file found inside
        /content/<target_folder>, preserving subdirectory traversal.

    Full scan mode (target_folder omitted or empty):
        Iterates the top-level entries of /content and downloads all
        files, recursing into subdirectories while skipping protected
        and caller-excluded folders.

        Protected folders (always skipped):
            .config, drive, .ipynb_checkpoints, sample_data

    Args:
        ignore_folders (list[str]): Folder names to skip during the full
                                    scan. Ignored when target_folder is set.
        target_folder  (str)      : Name of a specific subdirectory under
                                    /content to download from exclusively.
                                    Pass an empty string (default) to scan
                                    all of /content.

    Raises:
        ValueError: If a top-level entry under /content is neither a file
                    nor a directory (only raised in full scan mode).

    Notes:
        - Empty subdirectories are silently skipped (rglob yields no files).
        - files.download triggers one browser download dialog per file.
    """
    import shutil
    from pathlib import Path
    from google.colab import files

    protected = ['.config', 'drive', '.ipynb_checkpoints', 'sample_data']

    if target_folder:
        # --- Targeted mode: download everything inside the specified folder ---
        for p in Path('/content').joinpath(target_folder).rglob('*'):
            if p.is_file():
                files.download(p)
    else:
        # --- Full scan mode: download all user files under /content ---
        for p in Path('/content').glob('*'):
            if p.is_dir():
                if p.name in protected + ignore_folders:
                    continue  # Skip protected and user-excluded directories
                for pp in p.rglob('*'):
                    if pp.is_file():
                        files.download(pp)
            elif p.is_file():
                files.download(p)
            else:
                raise ValueError(f'It must be a file or directory: {p}')
