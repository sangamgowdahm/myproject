from src.exception import CustomException
import sys

if __name__ == "__main__":
    try:
        x = 1 / 0  # ZeroDivisionError
    except Exception as e:
        raise CustomException(e, sys)
