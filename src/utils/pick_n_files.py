"""Picks n files.

Picks n files randomly from a given directory and copies them to a new
directory.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

Created on:
    28 May 2020.
"""
from os import listdir
from pathlib import Path
from shutil import copyfile
from argparse import ArgumentParser
from random import sample
from utils.ask import ask
from tqdm import tqdm


def parse_args():
    p = ArgumentParser(description='picks n files randomly from a given '
                                   'directory and copies them to a new '
                                   'directory')

    p.add_argument('SRC', type=str, help='the source directory')
    p.add_argument('DEST', type=str, help='the output directory')
    p.add_argument('N', type=int, help='number of files to select')

    return p.parse_args()


def randomly_select_files(src_dir: Path, n: int) -> list:
    """Returns n filenames in the source directory that were sampled."""
    filenames = listdir(str(src_dir))
    indices = sample(list(range(len(filenames))), n)

    return [filenames[index] for index in indices]


def rmdir_with_contents(path_to_dir: Path):
    """Removes a directory and all its contents."""
    assert path_to_dir.exists(), 'Given path does not exist.'
    assert path_to_dir.is_dir(), 'Given path is not a directory.'

    for file in listdir(str(path_to_dir)):
        fp = path_to_dir / file
        if fp.is_file():
            fp.unlink()
        elif fp.is_dir():
            rmdir_with_contents(fp)
        elif fp.is_symlink():
            fp.unlink()

    path_to_dir.rmdir()


def main(src_dir: str, dest_dir: str, n: int):
    """Main loop which performs safety checks."""
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    if dest_dir.exists():
        if len(listdir(str(dest_dir))) > 0:
            if ask('Destination directory is not empty. Overwrite it?'):
                rmdir_with_contents(dest_dir)
                dest_dir.mkdir()

    files_to_copy = randomly_select_files(src_dir, n)
    for file in tqdm(files_to_copy):
        copyfile(src_dir / file, dest_dir / file)


if __name__ == '__main__':
    args = parse_args()
    main(args.SRC, args.DEST, args.N)
