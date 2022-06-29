import os
import sys
import time
from pathlib import Path

def test_version():
    import pydra
    print("pydra version: ", pydra.__version__)


if __name__ == '__main__':

    test_version()

    notebook_path = Path.cwd() / "notebooks"
    # Notebooks that should be tested
    notebooks = notebook_path.glob("*.ipynb")

    for test in notebooks:
        print("\n testing notebook: ", test)
        pytest_cmd = 'pytest --nbval-lax --nbval-cell-timeout 7200 -v -s %s' % test
        print(pytest_cmd)
        ex_code = os.system(pytest_cmd)
        if ex_code:
            raise Exception(f"notebook {test} fails")
