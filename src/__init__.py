import os
import sys

root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_path)
print(root_path)