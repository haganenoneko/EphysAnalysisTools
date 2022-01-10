# Copyright (c) 2022 Delbert Yip
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""Rename files from underescore-separated to TitleTitle format"""

import os 
from pathlib import Path 
from typing import List 
import logging 

def formatName(words: List[str]) -> str:
    words = [word.title() for word in words]
    return ''.join(words)

def rename_files(path: Path, ext: str='py'):
    """Rename files from 'a_b_c...' to AaBbCc"""
    
    if not path.is_dir(): 
        raise FileNotFoundError(path)
    
    logging.info(f"Searching path < {path} > for files with extension < {ext} >")
    files = path.glob(f"**/*.{ext}")
    
    if not files: 
        raise FileNotFoundError(f"No files with extension {ext} at directory: {path}")
    
    logging.info(files)
    
    for file in files:
        if '__init__' == file.stem: continue 
        
        new_name = formatName(file.stem.split('_'))
        new_name = f"{new_name}.{ext}"
        
        logging.info(new_name)
        
        new_path = file.parent.joinpath(new_name)
    
        try:
            file.rename(new_path)
        except Exception as e:
            logging.info(f"Could not rename: < {file} > to < {new_path} >")

logger = logging.basicConfig(level='DEBUG')

cwd = os.path.abspath(os.curdir)
rename_files(Path(f"{cwd}/src/GeneralProcess/"))
