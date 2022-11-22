"""
    Driver/main script runner. Executes the full process.
"""


from predict import Predicter


def main():
    pred = Predicter()
    pred.run_all()
    

if __name__ == "__main__":
    main()