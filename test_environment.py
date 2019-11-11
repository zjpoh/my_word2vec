import sys

def main():
    system_major = sys.version_info.major
    system_minor = sys.version_info.minor

    if (system_major, system_minor) >= (3, 7):
        raise TypeError(
            "This project requires Python {}.{}. Found: Python {}.{}".format(
                required_major, required_minor, system_major, system_minor))
    else:
        print(">>> Development environment passes all tests!")


if __name__ == '__main__':
    main()
