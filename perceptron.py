# Perceptron to classify letters in different fonts
import sys
import numpy as np
import random as r
import json

# Function to read in input file
def readFile(fName):
    # read in the first three numbers (input dimensions, out dimensions, num pairs)
    with open(fName) as inFile:
        inDim = inFile.read
# Prompt the user
def greet():
    print("Hello and welcome to my first neural network!")

# Menu to be displayed
def menu():
    print("\nPlease choose from one of the options below: ")
    print("1. Train using a training data file")
    print("2. Test using a trained weights file")
    print("3. Exit")

# Function to get the name of the file you want to train (EDIT)
def getTrainingFile():
    print("\nEnter the training data filename: ")

    try:
        trainingFile = input(">>> ")
        if trainingFile.endswith(".txt"): return trainingFile
        else:
            print("Please choose a valid text file")
            getTrainingFile()
    except ValueError:
        print("Please enter the name of the file you want to train")
        getTrainingFile()

# Function to ask user for intial weight values
def getWeights():
    print("\n0. To initialize weights to 0")
    print("1. To initialize weights to random values between -0.5 and 0.5")

    try:
        selection = int(input(">>> "))
        # Check if user inputs either 0 or 1
        if selection in range(0,2): return selection
        else:
            print("Please choose one of the two options: 0 or 1")
            getWeights()
    except ValueError:
        print("You need to choose an interger value (either 1 or 0)")
        getWeights()

# Function to ask user for number of epochs for program to run for
def getEpochs():
    print("\nEnter the max number of training epochs")
    try:
        epochs = int(input(">>> "))
        if epochs <= 0:
            print("You need to choose a number greater than 0")
            getEpochs()
        return epochs
    except ValueError:
        print("Please choose an integer value for the number of epochs")
        getEpochs()

'''
Function to prompt user for either a name to save
training results or testing results depending on the
selection option
'''
def getFileName(choice):
    # Getting name for trained weights
    if choice == 1:
        print("\nEnter a file name to save the trained weight values")
        trainWeight = input(">>> ")
        if trainWeight.endswith(".txt"): return trainWeight
        else:
            print("Please choose a filename that ends in .txt")
            getFileName(1)
    # Otherwise we're getting name to save testing reults to
    else:
        print("\nEnter a file name to save the testing results:")
        resultsFile = input(">>> ")
        if resultsFile.endswith(".txt"): return resultsFile
        else:
            print("Please choose a filename that ends in .txt")
            getFileName(2)

# Function to get the learning rate alpha from user
def getLearningRate():
    print("\nEnter the learning rate alpha from 0 to 1 (but not including 1)")

    try:
        rate = float(input(">>> "))
        if 0 <= rate < 1: return rate
        else:
            print("Your rate has to be either greater than or equal to 0" +
             " and less than 1")
            getLearningRate()
    except ValueError:
        print("Please choose a decimal number within the range")
        getLearningRate()

# Function to get the threshold theta value from user
def getThreshold():
    print("\nEnter a threshold for the theta (not too large or small)")

    try:
        theta = float(input(">>> "))
        if 0 < theta <= 0.1: return theta
        else:
            print("Please choose a theta greater than zero and less than (or equal to 0.1)")
            getThreshold()
    except ValueError:
        getThreshold()

# Function to get the name of the testing file
def getTestingFile():
    print("\nEnter the testing filename: ")

    try:
        testFile = input(">>> ")
        if testFile.endswith(".txt"): return testFile
        else:
            print("Please choose a valid text file")
            getTestingFile()
    except ValueError:
        print("Please enter the name of the file you want to train")
        getTestingFile()

# Function to call all the helper menu functions (UPDATE)
def callHelpers(option):
    try:
        if option == 1:
            trainFile = getTrainingFile()
            weights = getWeights()
            epochs = getEpochs()
            trainWFile = getFileName(option)
            learnRate = getLearningRate()
            threshold = getThreshold()
            return trainFile, weights, epochs, trainWFile, learnRate, threshold
        else:
            testFile = getTestingFile()
            trainResults = getFileName(option)
            # testResults = get
            return testFile, trainResults
            pass
    except KeyboardInterrupt:
        sys.exit()

# Function to read and store the values in the training file
def readAndStore(inFile):
    # Helper variables
    input_mode = True
    j = 0
    count = 1
    outs = -1
    inputs = open(inFile)

    # Store number of input dimensions/output dimensions/number of training pairs
    inDimensions = [int(val) for val in inputs.readline().split()]
    outDimensions = [int(val) for val in inputs.readline().split()]
    trainPairs = [int(val) for val in inputs.readline().split()]
    letVal = outDimensions[0] + 1 # to include the letter it corresponds to
    # create the list for inputs and the outputs
    trainP = [[None for x in range(0)] for x in range(trainPairs[0])]
    outP = [[None for x in range(0)] for x in range(trainPairs[0])]
    inputs.readline() # to skip over the blank line before training pairs
    contents = inputs.read() # To store remaining file

    # Read and store each training pair/output
    for item in contents.split():
        if input_mode and count >= 1:
            trainP[j].append(float(item))
            if count == inDimensions[0]: # finished getting training pairs
                input_mode = False # reset our input mode (so we're now doing outputs)
                count = -1 # so we don't mess with it
                outs = 1 # so we start on output next
                continue
            count+=1
            continue
        elif not input_mode and outs >= 1: # getting corresponding outputs
            outP[j].append(item)
            if outs == letVal: # finished our outputs
                input_mode = True # onto inputs
                count = 1
                outs = -1 # reset outs
                j+=1 # increment which training pair we're on
                continue
            outs+=1
            continue
    # close the file
    inputs.close()
    return inDimensions, outDimensions, trainPairs, trainP, outP

# Function to write the trained weights and bias to an output file
def writeWeights(finWeights, finBias, outFile, inputD):
    out = open(outFile, 'w')

    out.write(json.dumps(finWeights))
    out.write('\n')
    out.write(json.dumps(finBias))

    out.close()

    return outFile

# Function to create corresponding weight dictionary
def createWeights(inputD, outputD, weight):
    weights = {} # to store corresponding weights
    for num in range(inputD): # creates dict the size of input dimensions
        if not weight: # initialize weights to zero
            weights[num] = [weight] * outputD
        else: # initialize to random values
            weights[num] = [round(r.uniform(-0.5, 0.5), 2)] * outputD
    return weights

# Function to return the activation value
def activateF(y_in, theta):
    if y_in > theta: return 1
    elif -theta <= y_in <= theta: return 0
    else: return -1

# Functiont to call/run the perceptron algorithm
def perceptron(inDim, outDim, numPairs, trainVals, outputVals, weights, epochs, alpha, threshold, outFile):
    converged = False
    y_in = [0] * outDim # to store for each output
    y = [0] * outDim # activation for outputs
    summation = 0 # store summation for each t-pair
    update = 0
    era = 0
    neuronW = createWeights(inDim, outDim, weights) # Create weights
    bias = [1] * outDim # create bias

    while not converged:
        for index, pair in enumerate(trainVals):
            print("This is index: ", index)
            for j in range(outDim):
                for i, x in enumerate(pair):
                    summation += (x * neuronW[i][j])
                # Finished gathering the summation
                y_in[j] = round(bias[j] + summation, 2)
                y[j] = activateF(y_in[j], threshold) # Calculate y[j]
                print("My y[j] is: {} and my t is: {}".format(y[j], outputVals[index][j]))
                # Check to update bias and weights
                if float(outputVals[index][j]) != y[j]:
                    update = 1 # Something has been updated during the epoch
                    bias[j] += round(float(outputVals[index][j]) * alpha, 2)
                    for i, x in enumerate(pair):
                        neuronW[i][j] += round(float(outputVals[index][j]) * x * alpha, 2)
            summation = 0 # Reset summation for each training pair
        # Check if we've finished an epoch
        if index ==  numPairs-1:
            # Increment the eras
            era+=1
            print("This is the era we just finished: ", era)
            if era == epochs: # Check if we've done the max number of epochs allowed
                print("Training converged after {} epochs".format(era))
                converged = True
                return writeWeights(neuronW, bias, outFile, inDim)
                break
            if not update: # means we haven't changed anything in an epoch
                converged = True
                print("Training converged after {} epochs".format(era))
                return writeWeights(neuronW, bias, outFile, inDim)
                break
        update = 0 # Else just reset update

# Main program
def main():
    greet()
    while(1):
        menu()
        choice = int(input("\n>>> "))
        # user choose to train
        if choice == 1:
            # vals = callHelpers(choice)
            # Get variables out
            #trainFile, weights, epochs, outFile, learnRate, threshold = vals
            trainFile = getTrainingFile()
            outFile = getFileName(1)
            data = readAndStore(trainFile)
            # Get variables out
            inDim, outDim, pairs, trainP, outP = data
            perceptron(inDim[0], outDim[0], pairs[0], trainP, outP, 1, 20, 1, .5, outFile)
            # pass
        # user chooses to test
        elif choice == 2:


        # User selects any other option (change laterz)
        else: sys.exit()


# Run the program
if __name__ == '__main__':
    main()
