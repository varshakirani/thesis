import sys
sys.path.insert(0, '/Users/varshakirani/Documents/TUB/Thesis/imp/thesis/src/Utilities')

import tools






def main():
    print("NI Thesis")
    options = tools.parse_options()

    #Read Data and put it into panda datframe. Initially considering only means

    df = tools.data_extraction(options.data)

    if options.all:
        print("Running all the algorithms")

    if options.svm:
        print("Running SVM algorithm")

    if options.naive_bayes:
        print("Running naive bayes algorithm")




if __name__ == '__main__':
    main()

