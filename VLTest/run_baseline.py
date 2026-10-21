import sys

from run_vltest import main


if __name__ == "__main__":
    if "--method" not in sys.argv:
        sys.argv.extend(["--method", "SINVAD-VQ"])
    main()
