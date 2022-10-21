from dcgan import DCGAN
import sys

def main(arglist):
    """
	Run DCGAN.
	"""
    
    if len(arglist) < 2 or len(arglist) > 3:
        print("Bad argument count. Try again.")
        print("Usage: py driver.py train_dir test_dir optional_result_dir")
        return
    
    train_dir = arglist[0]
    test_dir = arglist[1]
    if len(arglist) == 3:
        result = arglist[2]
    else:
        result = "result\\"
    
    DCGAN(train_dir, test_dir, result)
    
	
if __name__ == '__main__':
    main(sys.argv[1:])

