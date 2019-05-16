#!/usr/bin/env python
import os
import sys

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
print("PATH IS ", (os.path.dirname(os.path.abspath(__file__))))
if __name__ == "__main__":
    os.environ["DJANGO_SETTINGS_MODULE"]= "simplifier.settings"

    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)
