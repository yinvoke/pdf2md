"""
Compatibility entrypoint.

This script now forwards to run_convert_example.py (the original convert test).
"""
from test.run_convert_example import main


if __name__ == "__main__":
    main()
