import pandas as pd 
import unittest 
import numpy as np 

from typing import List, Dict, Union, Any 

import sys, os 

cwd = os.path.abspath(os.curdir)
sys.path.insert(0, cwd + "/src/")

from GeneralProcess import Base
import tempfile, logging 
# ---------------------------------------------------------------------------- #

class TestLogging(unittest.TestCase):
        
    def test_empty_path(self):
        out = Base.createLogger(log_path=None)
        self.assertEqual(None, out)
        
    def test_instantiation(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            log_path = os.path.abspath(tmpdirname)
            
            log = Base.createLogger(log_path=log_path)
            self.assertIsNotNone(log)
            
            log.info("This is an info line")
            log.debug("This is a debug line")
            log.warning("This is a warning")
            log.error("This is an error")
                
if __name__ == '__main__':
    unittest.main()