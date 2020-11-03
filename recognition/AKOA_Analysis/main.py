from lateral_classification import train

def main():
    filelist = 'imgs/*.png'
    shape = (228, 260, 3)
    epochs = 10
    train(filelist, shape, epochs)

if __name__ == '__main__':
    main()



