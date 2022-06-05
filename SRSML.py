"""
Utilities for working with Omicron Scienta Matrix (MTRX) files
in conjunction pycroscopy (https://github.com/pycroscopy/pycroscopy) and
the associated h5USID file format. 
Steven R. Schofield, University College London, Jan. 2021
"""

# Ensure python 3 compatibility:
from __future__ import division, print_function, absolute_import, unicode_literals

# Operating system commands
import os  # this provides access to some operating system commands
import sys # provides access to some variables/functions used or maintained by the interpreter
import zipfile # tools to create, read, append, and list zip files.

# These two required for getting the creation date of the MTRX file and formatting it. 
import platform
import time

# additional required by sidpy etc?
from warnings import warn # Warning package in case something goes wrong
import subprocess # spawn subprocessed (apparently used by sidpy?)

# Numpy is for working with arrays:
import numpy as np

# manipulating paths (actually use this for dealing with the group names in the h5USID format)
import pathlib

# Scienta Omicron MTRX file loader
import access2thematrix
mtrx = access2thematrix.MtrxData()

# Packages for plotting:
import matplotlib.pyplot as plt

# This package for read/writing image files (jpg, tif, etc)
#from PIL import Image # python imaging library.  (I suspect this used for reading/writing jpegs etc)

# SPIEPy (Scanning Probe Image Enchanter using Python). Has useful image processing routines.s
import spiepy

# The package used for creating and manipulating HDF5 files:
import h5py

# Supporting package for pyUSID (storing, visualizing, and processing h5USID)
# Tutorials here: https://pycroscopy.github.io/sidpy/notebooks/00_basic_usage/index.html
import sidpy

# import pyUSID:
import pyUSID as usid
tran = usid.NumpyTranslator()
newResultsGroup = usid.hdf_utils.create_results_group
          

def create_h5USID_path(dataFilePath,imgFileName):
    """
    Create directory to store h5USID file, and return full file names for MTRX and h5USID files
    returns full path strings: imgFullFileName, h5FullFileName
    """
    
    # Use forward or backslash as appropriate for operating system
    dataFilePath=os.path.normpath(dataFilePath)
    imgFileName=os.path.normpath(imgFileName)
    
    currentDIR = os.getcwd()
    print("The current folder is: {}".format(currentDIR))
    
    # Make file name including full path (MTRX file)
    imgFullFileName=os.path.join(currentDIR,dataFilePath,imgFileName)
    
    print("Will attempt to open file: {}".format(imgFullFileName))
    
    dataFolderPath = os.path.dirname(os.path.dirname(dataFilePath))   
    # Create h5USID directory for writing the h5USID format image
    h5FilePath = os.path.join(dataFolderPath, "H5USID")
    os.makedirs(h5FilePath,exist_ok=True)
    
    print("Saving H5USID file in directory {}".format(h5FilePath))
    
    # Remove .Z_mtrx extension
    h5FileName = imgFileName[:-7]

    # Make h5USID file name including full path 
    h5FullFileName = os.path.join(h5FilePath, h5FileName + '.h5')
    
    print("Name of h5USID file saved: {}".format(h5FullFileName))
    
    return imgFullFileName, h5FullFileName


def MTRX_img_number(MTRXfileName):
    """
    Get the image number from the MTRX file name returns the image number as a string.  Will be something like "152_1"
    this splits the file whereever "--" occurs and we keep the last "piece"
    expecting a file name like: default_2017Jun09-162147_STM-STM_Spectroscopy--152_1.Z_mtrx
    """
    for word in MTRXfileName.split('--'):
        imgNumStr = word
    # get rid of the extension
    imgNumStr = imgNumStr.split(".")[0]
    return imgNumStr


def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last
            return stat.st_mtime
        

def MTRXtoh5USID(dataFilePath,imgFileName):
    """
    Read MTRX file and WRITE it to h5USID file
    """
    
    # create the h5USID folder and get full path strings for MTRX and h5USID files
    imgFullName, h5FullName = create_h5USID_path(dataFilePath,imgFileName)
    
    #############################
    # FIRST READ IN THE MTRX FILE
    #############################
    
    # Load MTRX file using the "access2thematrix" package; data and metadata will be stored in python object.
    imgBlock, message = mtrx.open(imgFullName)
    numTraces = len(imgBlock)
    
    # Print to screen the message that the file open was successful
    #print(message)
     
    # Initialise lists
    numRows = []
    numCols = []
    numPos = []
    specLength = []
    yQty = []
    yUnits = []
    yVec = []
    xQty = []
    xUnits = []
    xVec = []
    biasQty = []
    biasUnits = []
    biasVec = []
    mainDataName = []
    mainQty = []
    mainUnits = []
    imgNp = []
    imgNp1D = []

    # loop over the four images (FU, BU, FD, BD) and get the metadata
    for trace in range(numTraces):
        # get the trace data
        imgPy, message = mtrx.select_image(imgBlock[trace])
        # get number of rows and cols from loaded MTRX file
        numRows.append(imgPy.data.shape[0])
        numCols.append(imgPy.data.shape[1])
       
        # convert rows and cols lists to numpy arrays so can do arithmetic with the values
        numRowsArray = np.asarray(numRows)
        numColsArray = np.asarray(numCols)
        #
        numPos.append(numRowsArray[trace] * numColsArray[trace])
        numPosArray = np.asarray(numPos)
        # 
        yQty.append('Y')
        yUnits.append('m')
        yVec.append(np.linspace(0, imgPy.height, numRowsArray[trace], endpoint=True))
        # 
        xQty.append('X')
        xUnits.append('m')
        xVec.append(np.linspace(0, imgPy.width, numColsArray[trace], endpoint=True))
        #
        specLength.append(int(0))
        biasQty.append('None')
        biasUnits.append('None')
        biasVec.append([int(0)])
        #
        mainDataName.append(imgPy.channel_name_and_unit[0])   # "Z"
        mainQty.append(imgPy.channel_name_and_unit[0])
        mainUnits.append(imgPy.channel_name_and_unit[1])
        #
        imgTMP = np.asarray(imgPy.data)
        imgTMP1D = imgTMP.reshape(numPosArray[trace],1)
        #
        imgNp.append(imgTMP)
        imgNp1D.append(imgTMP1D)

    # Get the image number
    imgNumStr = MTRX_img_number(imgFullName)
    
    # This is the order that the traces are saved in the MTRX file. 
    traceNameList= ['Forward_Up', 'Backward_Up', 'Forward_Down', 'Backward_Down']
    imgNameList= ['m'+imgNumStr+'_FU', 'm'+imgNumStr+'_BU','m'+imgNumStr+'_FD','m'+imgNumStr+'_BD']

    # Now that the for loop is completed we make sure the number quantities are in numpy array format
    numRows = numRowsArray
    numCols = numColsArray
    numPos = numPosArray
    xVec = np.asarray(xVec)
    yVec = np.asarray(yVec)
    
    # Make sure the data is in the form of a numpy array and create a 1D array for writing to the h5USID file
    imgNp = np.asarray(imgNp)
    imgNp1D = np.asarray(imgNp1D)
    
    # Get the metadata - bias and current parameters from the MTRX file
    allParams = mtrx.get_experiment_element_parameters()
    allParams = allParams[0]
    
    # The MTRX file saves only forward and backward parameters, and does not differentiate between up and down
    for item in range(len(allParams)):
        # get forward and backward bias
        if allParams[item][0] == "GapVoltageControl.Voltage":
            biasForward = allParams[item][1]
        if allParams[item][0] == "GapVoltageControl.Alternate_Voltage":
            biasBackward = allParams[item][1]
        # get forward and backward current
        if allParams[item][0] == "Regulator.Setpoint_1":
            currentForward = allParams[item][1]
        if allParams[item][0] == "Regulator.Alternate_Setpoint_1":
            currentBackward = allParams[item][1]
    
    # format the meta data and make lists for writing in the output loops below
    currentForward = round(currentForward * 10 ** 12,2)
    currentBackward = round(currentBackward * 10 ** 12,2)
    biasForward = round(biasForward,2)
    biasBackward = round(biasBackward,2)
    imgBiasList = [biasForward, biasBackward, biasForward, biasBackward]
    imgCurrentList = [currentForward, currentBackward, currentForward, currentBackward]
    imgBiasUnits = "V"
    imgCurrentUnits = "pA"
    
    #############################
    # NOW WRITE h5USID FILE
    #############################
    
    # Delete the .h5 file if it exists
    if os.path.exists(h5FullName):
        os.remove(h5FullName)
    
    # Print to screen the name of the file to write
    print("Writing h5 file: ")
    print(h5FullName)
    
    # Create empty h5 file and internal groups and subgroups
    h5file = h5py.File(h5FullName,"a")

    # Create the top level group for the STM topography data 
    h5TopoGroup = usid.hdf_utils.create_indexed_group(h5file, 'Topography') # contains the STM data
    
    # Write each of the images (forward up, backward up, etc) to a new group in the h5USID file
    for trace in range(numTraces):

        # Create the subgroup for each of the traces. 
        h5TraceGroup = usid.hdf_utils.create_indexed_group(h5TopoGroup, traceNameList[trace]) # for the raw STM image
        
        # Define the dimensions of the data for writing the h5USID format data.
        posDims = [usid.Dimension(xQty[trace], xUnits[trace], xVec[trace]),
                   usid.Dimension(yQty[trace], yUnits[trace], yVec[trace])]
        specDims = usid.Dimension(biasQty[trace], biasUnits[trace], biasVec[trace])
        
        # Write the data to the h5 file
        h5main = usid.hdf_utils.write_main_dataset(h5TraceGroup,  # parent HDF5 group
                                               imgNp1D[trace],  # shape of Main dataset
                                               imgNameList[trace],  # Name of main dataset
                                               mainDataName[trace],  # Physical quantity contained in Main dataset
                                               mainUnits[trace],  # Units for the physical quantity
                                               posDims,  # Position dimensions
                                               specDims,  # Spectroscopic dimensions
                                               dtype=imgNp1D[trace].dtype,  # data type / precision
                                               compression='gzip')
        
        # Some more meta data information on the MTRX file that we will store in the h5USID file
        dataFilePath = os.path.split(h5FullName)[0]
        imgFileName = os.path.split(h5FullName)[1]
        MTRXcreationDate = creation_date(h5FullName)
        MTRXcreationDateString = time.ctime(MTRXcreationDate)
        
        # Write the meta data
        sidpy.hdf_utils.write_simple_attrs(h5TraceGroup, {'Original File': imgFileName,
                                                 'MTRX file creation date string': MTRXcreationDateString,
                                                 'MTRX file creation date': MTRXcreationDate,
                                                 'MTRX File Path': dataFilePath,
                                                 'Image Number': imgNumStr,
                                                 'Trace Name': traceNameList[trace],                                         
                                                 'Image Bias': imgBiasList[trace],
                                                 'Image Current': imgCurrentList[trace],
                                                 'Bias Units': imgBiasUnits,
                                                 'Current Units': imgCurrentUnits})
        usid.hdf_utils.write_book_keeping_attrs(h5TopoGroup)
        
    h5file.close()   
    return h5FullName
    
    
def h5_show_groups(h5FullName):
    """
    Print out a group list
    """
      
    # Load the h5USID file
    h5file = h5py.File(h5FullName, mode='r')
    
    structure = sidpy.hdf_utils.print_tree(h5file, rel_paths=True)
    
    # Close the file.
    h5file.close()
    
    

def h5_show_metadata(h5FullName):
    """
    Print out the metadata
    """
    
    # Load the h5USID file
    h5file = h5py.File(h5FullName, mode='r')
    
    # Define the topography group (this is expecting the file to have been created using
    # MTRXtoh5USID(MTRXfileName,h5USIDfileName), but might work with others...?
    h5TopoGroupName = list(h5file.keys())[0]
    
    h5TopoGroup = h5file[h5TopoGroupName]
    
    # Print to screen the meta data of the file.  
    for key, val in sidpy.hdf_utils.get_attributes(h5TopoGroup).items():
        print('{} : {}'.format(key, val))
    
    # Close the file.
    h5file.close()

    
def h5_show_all_metadata(h5FullName):
    """
    Print out all metadata
    """
    
    # Load the h5USID file
    h5file = h5py.File(h5FullName, mode='r')
    
    # Define the topography group (this is expecting the file to have been created using
    # MTRXtoh5USID(MTRXfileName,h5USIDfileName), but might work with others...?
    h5TopoGroupName = list(h5file.keys())[0]

    # Get the main datasets
    mainDataSets = usid.hdf_utils.get_all_main(h5file) # Finds all the main data sets. 
    numTraces = len(mainDataSets)
    
    h5TopoGroup = h5file[h5TopoGroupName]
    
    print('---------------------------------')
    print(h5TopoGroupName)
    print('---------------------------------')
        
    # Print to screen the meta data of the file - the top level group.
    for key, val in sidpy.hdf_utils.get_attributes(h5TopoGroup).items():
        print('{} : {}'.format(key, val))
    
    traceFullNameList = []  
    for trace in range(numTraces):
        traceFullNameList.append(mainDataSets[trace].name)
            
    traceGroupNameList = []
    traceNameList = []
    for trace in range(numTraces):
        imgGroupName = pathlib.PurePath(traceFullNameList[trace]).parent
        imgName = pathlib.PurePath(traceFullNameList[trace]).name
        traceGroupNameList.append(str(imgGroupName))
        traceNameList.append(str(imgName))
        traceGroup = h5file[traceGroupNameList[trace]]
        print('---------------------------------')
        print(traceGroupNameList[trace])
        print(traceNameList[trace])
        print('---------------------------------')
        # Print to screen the meta data of the file - the top level group.
        for key, val in sidpy.hdf_utils.get_attributes(traceGroup).items():
            print('{} : {}'.format(key, val))
            
    # Close the file.
    h5file.close()

def h5_get_metadata_parameter(h5FullName,paramName):
    """
    Get values for some metadata parameter
    """
    
    # Load the h5USID file
    h5file = h5py.File(h5FullName, mode='r')
  
    # Get the main datasets
    mainDataSets = usid.hdf_utils.get_all_main(h5file) # Finds all the main data sets. 
    numTraces = len(mainDataSets)
    
    # Define the topography group (this is expecting the file to have been created using
    # MTRXtoh5USID(MTRXfileName,h5USIDfileName), but might work with others...?
    TopoGroupName = list(h5file.keys())[0]
    
  #  TopoGroup = h5file[TopoGroupName]
  #  for key, val in sidpy.hdf_utils.get_attributes(TopoGroup).items():
  #      print('{} : {}'.format(key, val))
    
    traceFullNameList = []  
    for trace in range(numTraces):
        traceFullNameList.append(mainDataSets[trace].name)
    
    traceGroupNameList = []
    returnList = []
    for trace in range(numTraces):
        imgGroupName = pathlib.PurePosixPath(traceFullNameList[trace]).parent
        traceGroupNameList.append(str(imgGroupName))
        traceGroup = h5file[traceGroupNameList[trace]]
        for key, val in sidpy.hdf_utils.get_attributes(traceGroup).items():
            if key == paramName:
                returnList.append(val)
    
    return returnList
    # Close the file.
    h5file.close()
    
    

def h5_get_metadata_parameters_for_image_title(h5FullName):
    """
    Get values for some metadata parameter
    """
    
#    print("DEBUG2: h5FullName: {}".format(h5FullName))
    
    # Load the h5USID file
    h5file = h5py.File(h5FullName, mode='r')
  
    # Get the main datasets
    mainDataSets = usid.hdf_utils.get_all_main(h5file) # Finds all the main data sets. 
    numTraces = len(mainDataSets)
    
#    print("DEBUG2: numTraces: {}".format(numTraces))
    
    # Get name of top level group (should be 'Topography_000')
    TopoGroupName = list(h5file.keys())[0]
    
#    print("DEBUG2: TopoGroupName: {}".format(TopoGroupName))
    
  #  TopoGroup = h5file[TopoGroupName]
  #  for key, val in sidpy.hdf_utils.get_attributes(TopoGroup).items():
  #      print('{} : {}'.format(key, val))
    
    traceFullNameList = []  
    for trace in range(numTraces):
        traceFullNameList.append(mainDataSets[trace].name)  
    
#    print("DEBUG2: traceFullNameList: {}".format(traceFullNameList))
    
    traceGroupNameList = []
    traceNameList = []
    dateList = []
    biasList = []
    currentList = []
    biasUnitsList = []
    currentUnitsLists = []
    
    for trace in range(numTraces):
#        print("DEBUG2: traceFullNameList[trace]: {}".format(traceFullNameList[trace]))
        imgGroupName = pathlib.PurePosixPath(traceFullNameList[trace]).parent
#        print("DEBUG2: imgGroupName: {}".format(imgGroupName))
        
        traceGroupNameList.append(str(imgGroupName))
        traceGroup = h5file[traceGroupNameList[trace]]
        imgName = pathlib.PurePosixPath(traceFullNameList[trace]).name
        
        traceNameList.append(str(imgName))
        for key, val in sidpy.hdf_utils.get_attributes(traceGroup).items():
            if key == 'MTRX file creation date string':
                dateList.append(val)
            elif key == 'Image Bias':
                biasList.append(val)
            elif key == 'Image Current':
                currentList.append(val)
            elif key == 'Bias Units':
                biasUnitsList.append(val)
            elif key == 'Current Units':
                currentUnitsLists.append(val)
                
#    print("DEBUG2: traceGroupNameList: {}".format(traceGroupNameList))
    
    return traceGroupNameList, traceNameList, dateList, biasList, currentList, biasUnitsList, currentUnitsLists
    # Close the file.
    h5file.close()


def h5_get_xyVec(h5FullName):
    """
    get xVec and yVec data
    """
    
    h5file = h5py.File(h5FullName, mode='r')
    
    # Get the name of the top level group (will be Topography_000 if created with SPMPy)
    h5TopoGroupName = list(h5file.keys())[0]
    
    # Get the h5 group of the top level group. 
    h5TopoGroup = h5file[h5TopoGroupName]

    # Get all the main data sets in this group. 
    mainDataSets = usid.hdf_utils.get_all_main(h5TopoGroup)
    
    yValList = []
    xValList = []
    xUnitList = []
    yUnitList = []
    counter = 0
    for trace in range(len(mainDataSets)):
        
        #Select the trace data set 
        h5main = mainDataSets[trace]
        h5img = h5TopoGroup[h5main.name]
            
        dsetList = sidpy.hdf_utils.get_auxiliary_datasets(h5img, ['Position_Indices', 'Position_Values',
                                                                   'Spectroscopic_Indices', 'Spectroscopic_Values'])
        posInds, posVals, specInds, specVals = dsetList
        posUnitValues = usid.hdf_utils.get_unit_values(posInds, posVals)
        yVals, xVals = posUnitValues.values()
        
        
   #     print("h5_get_xyVec(h5FullName)")
   #     print("trace ", trace)
   #     print("len(xVals)", len(xVals))
   #     print("len(yVals)", len(yVals))
        yValList.append(yVals)
        xValList.append(xVals)
        
        posUnitItems = sidpy.hdf_utils.get_attributes(h5main.h5_pos_inds).items()
        for key, val in posUnitItems:
            if key=="units":
                yUnit, xUnit = val
        yUnitList.append(yUnit)
        xUnitList.append(xUnit)
        
        
    #   pos_dim_sizes = usid.hdf_utils.get_dimensionality(posInds)
    #   pos_dim_names = sidpy.hdf_utils.get_attr(posInds, 'labels')

    #   for name, length in zip(pos_dim_names, pos_dim_sizes):
    #       print('{} : {}'.format(name, length))

    #   for key, val in posUnitValues.items():
    #      print('{} : {}'.format(key, val))

    # Close the file.
    h5file.close()
    return yValList, xValList, yUnitList, xUnitList
    
    

def h5_getNpImg(h5FullName, **kwargs):
    """
    This function loads on h5USID file and provides the top image as a numpy
    array.  Additional images are ignored.
    """
    
    # Read the h5USID file from disk
    h5file = h5py.File(h5FullName, mode='r')

    # Get the main datasets
    h5main = usid.hdf_utils.get_all_main(h5file) # Finds all the main data sets. 
    numTraces = len(h5main)
    if numTraces > 1:
        print("Warning: multiple images detected.  Taking only the first image.")

    # Get the axes from the auxilliary data
    yVec, xVec, yUnits, xUnits = h5_get_xyVec(h5FullName)  # note these are lists of arrays depending on how many images in the file

    # Get name of top level group (should be 'Topography_000'). We need this later for copying the meta data to the new file
    h5TopoGroupName = list(h5file.keys())[0]
    h5TopoGroup = h5file[h5TopoGroupName]

    # read data from the original file
    traceGroupNameList, imgNameList, dateList, biasList, currentList, biasUnitsList, currentUnitsLists = h5_get_metadata_parameters_for_image_title(h5FullName)

    traceName = pathlib.PurePosixPath(traceGroupNameList[0]).name

    # extract the image to a numpy array
    imgNp = h5main[0].get_n_dim_form().squeeze()
    
    return yVec[0], xVec[0], yUnits[0], xUnits[0], imgNp
    #return yVec, xVec, yUnits, xUnits, imgNp

    h5file.close()

    

def h5_manualSave(h5FullName, descrip, img):
    """
    This function allows to save a manually changed img.  It loads in the original
    h5USID image to obtain all relevant parameters.  Most of this code is copied
    from the manipulate function.  At the moment this assumes only one image.
    """
    
    imgNp = [img]  # could change this later if wanted to allow editing of more than one image. 
    selectone = "Unknown" # setting this to something other than "None" is sufficient that only one image is saved. 
    imgSelect = 0
    
    #############################
    # FIRST READ h5USID FILE
    #############################
    
    # Read the h5USID file from disk
    h5file = h5py.File(h5FullName, mode='r')
    
    # Get the main datasets
    h5main = usid.hdf_utils.get_all_main(h5file) # Finds all the main data sets. 
    numTraces = len(h5main)
    
    # Get the axes from the auxilliary data
    yVec, xVec, yUnits, xUnits = h5_get_xyVec(h5FullName)
    
    # Get name of top level group (should be 'Topography_000'). We need this later for copying the meta data to the new file
    h5TopoGroupName = list(h5file.keys())[0]
    h5TopoGroup = h5file[h5TopoGroupName]
    
    # read data from the original file
    traceGroupNameList, imgNameList, dateList, biasList, currentList, biasUnitsList, currentUnitsLists = \
    h5_get_metadata_parameters_for_image_title(h5FullName)

    traceNameList = []
    for trace in range(numTraces):
        traceName = pathlib.PurePath(traceGroupNameList[trace]).name
        traceNameList.append(str(traceName[:-4]))
        
    # Expecting traceNameList to be
    # traceNameList= ['Backward_Down','Backward_Up','Forward_Down','Forward_Up']

    
    # extract the image frames to a numpy array (will be 4x images or 2x images)
 #   imgNp = []
 #   for trace in range(numTraces):
 #       imgTraceNp = h5main[trace].get_n_dim_form().squeeze()
 #       imgNp.append(imgTraceNp)
   
      
    #############################
    #  WRITE h5USID FILE
    #############################

    # Now create a new h5USID file for the manipulated data. 
    h5mFullName = h5FullName[:-3]+descrip+".h5"

    # Delete the .h5 file if it exists
    if os.path.exists(h5mFullName):
        os.remove(h5mFullName)
    
    print("Writing h5 file: ")
    print(h5mFullName)
    # Create empty h5 file and internal groups and subgroups
    h5mfile = h5py.File(h5mFullName,"a")

    # NEED TO CHANGE THIS.  THESE SHOULD BE READ FROM THE FILE.
    xQty = ['X', 'X', 'X', 'X']
    yQty = ['Y', 'Y', 'Y', 'Y']
    mainUnits = ['m', 'm', 'm', 'm']
    mainDataName = ['Z', 'Z', 'Z', 'Z']
    
    # Since we are dealing only with image data, the spectroscopy component here is all empty. 
    specQtyList = ['None', 'None', 'None', 'None']
    specUnitsList = ['None', 'None', 'None', 'None']
    specVecList = [[int(0)], [int(0)], [int(0)], [int(0)]]
    
    # Need to convert 2D image   data to 1D for writing to h5USID
    imgNp1D = []
    for trace in range(numTraces):
        imgTraceNp = imgNp[trace]
        numPos = imgTraceNp.shape[0] * imgTraceNp.shape[1]
        imgTrace1D = imgTraceNp.reshape(numPos,1)
        imgNp1D.append(imgTrace1D)
        
    # Create the top level group for the STM topography data 
    h5mTopoGroup = usid.hdf_utils.create_indexed_group(h5mfile, 'Topography') # contains the STM data
    
    # Write each of the images (forward up, backward up, etc) to a new group in the h5USID file
    h5mTraceGroupList = []
    for trace in range(numTraces):

        # Create the subgroup for each of the traces. 
        h5mTraceGroup = usid.hdf_utils.create_indexed_group(h5mTopoGroup, traceNameList[trace]) # for the raw STM image
        h5mTraceGroupList.append(h5mTraceGroup)
        # Define the dimensions of the data for writing the h5USID format data.
        posDims = [usid.Dimension(xQty[trace], xUnits[trace], xVec[trace]),
                   usid.Dimension(yQty[trace], yUnits[trace], yVec[trace])]
        
        specDims = usid.Dimension(specQtyList[trace], specUnitsList[trace], specVecList[trace])

        # Get the name of the top level group (will be Topography_000 if created with SPMPy)
        #h5TopoGroupName = list(h5file.keys())[0]

        # Get the h5 group of the top level group. 
        h5TopoGroup = h5file[h5TopoGroupName]
        
        # Write the data to the h5 file
        h5main = usid.hdf_utils.write_main_dataset(h5mTraceGroup,  # parent HDF5 group
                                               imgNp1D[trace],  # 
                                               imgNameList[trace],  # Name of main dataset
                                               mainDataName[trace],  # Physical quantity contained in Main dataset
                                               mainUnits[trace],  # Units for the physical quantity
                                               posDims,  # Position dimensions
                                               specDims,  # Spectroscopic dimensions
                                               dtype=imgNp1D[trace].dtype,  # data type / precision
                                               compression='gzip')
            
    # This command copies all the meta data from the original file to the new file
    sidpy.hdf.hdf_utils.copy_attributes(h5TopoGroup, h5mTopoGroup)
    
    if selectone == None:
        # Copy the meta data for each of the image groups as well (this is the important step)
        for trace in range(numTraces):
            traceGroup = h5file[traceGroupNameList[trace]]
            traceMGroup = h5mfile[traceGroupNameList[trace]]
            sidpy.hdf.hdf_utils.copy_attributes(traceGroup,traceMGroup)
    elif selectone != None:
        # Write the meta data
        imgFileName = h5_get_metadata_parameter(h5FullName,'Original File')[imgSelect]
        MTRXcreationDateString = h5_get_metadata_parameter(h5FullName,'MTRX file creation date string')[imgSelect]
        MTRXcreationDate = h5_get_metadata_parameter(h5FullName,'MTRX file creation date')[imgSelect]
        dataFilePath = h5_get_metadata_parameter(h5FullName,'MTRX File Path')[imgSelect]
        imgNumStr = h5_get_metadata_parameter(h5FullName,'Image Number')[imgSelect]
        traceName = h5_get_metadata_parameter(h5FullName,'Trace Name')[imgSelect]
        imgBias = h5_get_metadata_parameter(h5FullName,'Image Bias')[imgSelect]
        imgCurrent = h5_get_metadata_parameter(h5FullName,'Image Current')[imgSelect]
        imgBiasUnits = h5_get_metadata_parameter(h5FullName,'Bias Units')[imgSelect]
        imgCurrentUnits = h5_get_metadata_parameter(h5FullName,'Current Units')[imgSelect]
        sidpy.hdf_utils.write_simple_attrs(h5mTraceGroup, {'Original File': imgFileName,
                                                 'MTRX file creation date string': MTRXcreationDateString,
                                                 'MTRX file creation date': MTRXcreationDate,
                                                 'MTRX File Path': dataFilePath,
                                                 'Image Number': imgNumStr,
                                                 'Trace Name': traceName,                                         
                                                 'Image Bias': imgBias,
                                                 'Image Current': imgCurrent,
                                                 'Bias Units': imgBiasUnits,
                                                 'Current Units': imgCurrentUnits})

    h5file.close()
    h5mfile.close()    
    return h5mFullName





  
def h5_manualSaveNewSize(h5FullName, descrip, xVecNew, yVecNew, img):
    """
    This function allows to save a manually changed img.  It loads in the original
    h5USID image to obtain all relevant parameters.  Most of this code is copied
    from the manipulate function.  At the moment this assumes only one image.
    """
    
    imgNp = [img]  # could change this later if wanted to allow editing of more than one image. 
    selectone = "Unknown" # setting this to something other than "None" is sufficient that only one image is saved. 
    imgSelect = 0
    
    #############################
    # FIRST READ h5USID FILE
    #############################
    
    # Read the h5USID file from disk
    h5file = h5py.File(h5FullName, mode='r')
    
    # Get the main datasets
    h5main = usid.hdf_utils.get_all_main(h5file) # Finds all the main data sets. 
    numTraces = len(h5main)
    
    # Get the axes from the auxilliary data
    yVec, xVec, yUnits, xUnits = h5_get_xyVec(h5FullName)
    
    # Change xVec and yVec to the new values
    yVec = []
    xVec = []
    yVec.append(yVecNew)
    xVec.append(xVecNew)
    
    # Get name of top level group (should be 'Topography_000'). We need this later for copying the meta data to the new file
    h5TopoGroupName = list(h5file.keys())[0]
    h5TopoGroup = h5file[h5TopoGroupName]
    
    # read data from the original file
    traceGroupNameList, imgNameList, dateList, biasList, currentList, biasUnitsList, currentUnitsLists = \
    h5_get_metadata_parameters_for_image_title(h5FullName)

    traceNameList = []
    for trace in range(numTraces):
        traceName = pathlib.PurePath(traceGroupNameList[trace]).name
        traceNameList.append(str(traceName[:-4]))
        
    # Expecting traceNameList to be
    # traceNameList= ['Backward_Down','Backward_Up','Forward_Down','Forward_Up']

    
    # extract the image frames to a numpy array (will be 4x images or 2x images)
 #   imgNp = []
 #   for trace in range(numTraces):
 #       imgTraceNp = h5main[trace].get_n_dim_form().squeeze()
 #       imgNp.append(imgTraceNp)
   
      
    #############################
    #  WRITE h5USID FILE
    #############################

    # Now create a new h5USID file for the manipulated data. 
    h5mFullName = h5FullName[:-3]+descrip+".h5"

    # Delete the .h5 file if it exists
    if os.path.exists(h5mFullName):
        os.remove(h5mFullName)
    
    print("Writing h5 file: ")
    print(h5mFullName)
    # Create empty h5 file and internal groups and subgroups
    h5mfile = h5py.File(h5mFullName,"a")

    # NEED TO CHANGE THIS.  THESE SHOULD BE READ FROM THE FILE.
    xQty = ['X', 'X', 'X', 'X']
    yQty = ['Y', 'Y', 'Y', 'Y']
    mainUnits = ['m', 'm', 'm', 'm']
    mainDataName = ['Z', 'Z', 'Z', 'Z']
    
    # Since we are dealing only with image data, the spectroscopy component here is all empty. 
    specQtyList = ['None', 'None', 'None', 'None']
    specUnitsList = ['None', 'None', 'None', 'None']
    specVecList = [[int(0)], [int(0)], [int(0)], [int(0)]]
    
    # Need to convert 2D image   data to 1D for writing to h5USID
    imgNp1D = []
    for trace in range(numTraces):
        imgTraceNp = imgNp[trace]
        numPos = imgTraceNp.shape[0] * imgTraceNp.shape[1]
        imgTrace1D = imgTraceNp.reshape(numPos,1)
        imgNp1D.append(imgTrace1D)
        
    # Create the top level group for the STM topography data 
    h5mTopoGroup = usid.hdf_utils.create_indexed_group(h5mfile, 'Topography') # contains the STM data
    
    # Write each of the images (forward up, backward up, etc) to a new group in the h5USID file
    h5mTraceGroupList = []
    for trace in range(numTraces):

        # Create the subgroup for each of the traces. 
        h5mTraceGroup = usid.hdf_utils.create_indexed_group(h5mTopoGroup, traceNameList[trace]) # for the raw STM image
        h5mTraceGroupList.append(h5mTraceGroup)
        # Define the dimensions of the data for writing the h5USID format data.
        posDims = [usid.Dimension(xQty[trace], xUnits[trace], xVec[trace]),
                   usid.Dimension(yQty[trace], yUnits[trace], yVec[trace])]
        
        specDims = usid.Dimension(specQtyList[trace], specUnitsList[trace], specVecList[trace])

        # Get the name of the top level group (will be Topography_000 if created with SPMPy)
        #h5TopoGroupName = list(h5file.keys())[0]

        # Get the h5 group of the top level group. 
        h5TopoGroup = h5file[h5TopoGroupName]
        
        # Write the data to the h5 file
        h5main = usid.hdf_utils.write_main_dataset(h5mTraceGroup,  # parent HDF5 group
                                               imgNp1D[trace],  # 
                                               imgNameList[trace],  # Name of main dataset
                                               mainDataName[trace],  # Physical quantity contained in Main dataset
                                               mainUnits[trace],  # Units for the physical quantity
                                               posDims,  # Position dimensions
                                               specDims,  # Spectroscopic dimensions
                                               dtype=imgNp1D[trace].dtype,  # data type / precision
                                               compression='gzip')
            
    # This command copies all the meta data from the original file to the new file
    sidpy.hdf.hdf_utils.copy_attributes(h5TopoGroup, h5mTopoGroup)
    
    if selectone == None:
        # Copy the meta data for each of the image groups as well (this is the important step)
        for trace in range(numTraces):
            traceGroup = h5file[traceGroupNameList[trace]]
            traceMGroup = h5mfile[traceGroupNameList[trace]]
            sidpy.hdf.hdf_utils.copy_attributes(traceGroup,traceMGroup)
    elif selectone != None:
        # Write the meta data
        imgFileName = h5_get_metadata_parameter(h5FullName,'Original File')[imgSelect]
        MTRXcreationDateString = h5_get_metadata_parameter(h5FullName,'MTRX file creation date string')[imgSelect]
        MTRXcreationDate = h5_get_metadata_parameter(h5FullName,'MTRX file creation date')[imgSelect]
        dataFilePath = h5_get_metadata_parameter(h5FullName,'MTRX File Path')[imgSelect]
        imgNumStr = h5_get_metadata_parameter(h5FullName,'Image Number')[imgSelect]
        traceName = h5_get_metadata_parameter(h5FullName,'Trace Name')[imgSelect]
        imgBias = h5_get_metadata_parameter(h5FullName,'Image Bias')[imgSelect]
        imgCurrent = h5_get_metadata_parameter(h5FullName,'Image Current')[imgSelect]
        imgBiasUnits = h5_get_metadata_parameter(h5FullName,'Bias Units')[imgSelect]
        imgCurrentUnits = h5_get_metadata_parameter(h5FullName,'Current Units')[imgSelect]
        sidpy.hdf_utils.write_simple_attrs(h5mTraceGroup, {'Original File': imgFileName,
                                                 'MTRX file creation date string': MTRXcreationDateString,
                                                 'MTRX file creation date': MTRXcreationDate,
                                                 'MTRX File Path': dataFilePath,
                                                 'Image Number': imgNumStr,
                                                 'Trace Name': traceName,                                         
                                                 'Image Bias': imgBias,
                                                 'Image Current': imgCurrent,
                                                 'Bias Units': imgBiasUnits,
                                                 'Current Units': imgCurrentUnits})

    h5file.close()
    h5mfile.close()    
    return h5mFullName




  
def h5_manipulate(h5FullName, **kwargs):
    """
    This function loads on h5USID file, does something to the data
    then saves that data in a modified h5USID file with different name.
    """
    background = kwargs.get('background', None)
    offset = kwargs.get('offset', None)
    selectone = kwargs.get('selectone', None)
    
#    print("DEBUG: h5FullName: {}".format(h5FullName))
    #############################
    # FIRST READ h5USID FILE
    #############################
    
    # Read the h5USID file from disk
    h5file = h5py.File(h5FullName, mode='r')
    
    # Get the main datasets
    h5main = usid.hdf_utils.get_all_main(h5file) # Finds all the main data sets. 
    numTraces = len(h5main)
    
    # Get the axes from the auxilliary data
    yVec, xVec, yUnits, xUnits = h5_get_xyVec(h5FullName)
    
    # Get name of top level group (should be 'Topography_000'). We need this later for copying the meta data to the new file
    h5TopoGroupName = list(h5file.keys())[0]
    h5TopoGroup = h5file[h5TopoGroupName]
    
#    print("DEBUG: h5TopoGroupName: {}".format(h5TopoGroupName))
#    print("DEBUG: h5TopoGroup: {}".format(h5TopoGroup))
    
    # read data from the original file
    traceGroupNameList, imgNameList, dateList, biasList, currentList, biasUnitsList, currentUnitsLists = \
    h5_get_metadata_parameters_for_image_title(h5FullName)

#    print("DEBUG: traceGroupNameList: {}".format(traceGroupNameList))
#    print("DEBUG: imgNameList: {}".format(imgNameList))
    
    traceNameList = []
    for trace in range(numTraces):
        traceName = pathlib.PurePosixPath(traceGroupNameList[trace]).name
        traceNameList.append(str(traceName[:-4]))
        
    # Expecting traceNameList to be
    # traceNameList= ['Backward_Down','Backward_Up','Forward_Down','Forward_Up']

    
    # extract the image frames to a numpy array (will be 4x images or 2x images)
    imgNp = []
    for trace in range(numTraces):
        imgTraceNp = h5main[trace].get_n_dim_form().squeeze()
        imgNp.append(imgTraceNp)
   
    
    #############################
    # NOW PERFORM MANIPULATION
    #############################
    
    # create extensions to add to filename to indicate the manipulation that was made
    manipExt = ""
    if background == "plane":
        manipExt = manipExt+"p" # order 1 polynomial background
        # Use SPIEpy package to plane subtract the image
        for trace in range(numTraces):
            imgNp[trace] = spiepy.flatten_poly_xy(imgNp[trace], mask=None, deg=1)[0]
            imgNp[trace] = imgNp[trace] - np.min(imgNp[trace])
    elif background == "plane2": # order 2 polynomial background
        manipExt = manipExt+"p2"
        # Use SPIEpy package to plane subtract the image
        for trace in range(numTraces):
            imgNp[trace] = spiepy.flatten_poly_xy(imgNp[trace], mask=None, deg=2)[0]
            imgNp[trace] = imgNp[trace] - np.min(imgNp[trace])
    elif background == "mask": # order 2 polynomial background
        manipExt = manipExt+"m"
        # Use SPIEpy package to plane subtract the image
        for trace in range(numTraces): 
            imgNp[trace], mask, n = spiepy.flatten_by_iterate_mask(imgNp[trace])
            imgNp[trace] = imgNp[trace] - np.min(imgNp[trace])
    elif background == "planebackground":
        manipExt = manipExt+"b" # order 1 polynomial background; get the background
        # Use SPIEpy package to plane subtract the image
        for trace in range(numTraces):
            imgNp[trace] = spiepy.flatten_poly_xy(imgNp[trace], mask=None, deg=1)[1]
            imgNp[trace] = imgNp[trace] - np.min(imgNp[trace])
    elif background == "plane2background": # order 2 polynomial background
        manipExt = manipExt+"b2"
        # Use SPIEpy package to plane subtract the image; get the background
        for trace in range(numTraces):
            imgNp[trace] = spiepy.flatten_poly_xy(imgNp[trace], mask=None, deg=2)[1]
            imgNp[trace] = imgNp[trace] - np.min(imgNp[trace])
    elif selectone == "FU":
        manipExt = manipExt+"FU" # order 1 polynomial background
        imgSelect = 3
    elif selectone == "BU":
        manipExt = manipExt+"FU" # order 1 polynomial background
        imgSelect = 1
    elif selectone == "FD":
        manipExt = manipExt+"FU" # order 1 polynomial background
        imgSelect = 2
    elif selectone == "BD":
        manipExt = manipExt+"FU" # order 1 polynomial background
        imgSelect = 0
    
    if selectone != None:
        numTraces = 1
        imgNp[0] = imgNp[imgSelect]
        xVec[0] = xVec[imgSelect]
        yVec[0] = yVec[imgSelect]
        traceNameList[0] = traceNameList[imgSelect]
        imgNameList[0] = imgNameList[imgSelect]
    
    #############################
    # NOW WRITE NEW h5USID FILE
    #############################

    # Now create a new h5USID file for the manipulated data. 
    h5mFullName = h5FullName[:-3]+manipExt+".h5"

    # Delete the .h5 file if it exists
    if os.path.exists(h5mFullName):
        os.remove(h5mFullName)
    
    print("Writing h5 file: ")
    print(h5mFullName)
    # Create empty h5 file and internal groups and subgroups
    h5mfile = h5py.File(h5mFullName,"a")

    # NEED TO CHANGE THIS.  THESE SHOULD BE READ FROM THE FILE.
    xQty = ['X', 'X', 'X', 'X']
    yQty = ['Y', 'Y', 'Y', 'Y']
    mainUnits = ['m', 'm', 'm', 'm']
    mainDataName = ['Z', 'Z', 'Z', 'Z']
    
    # Since we are dealing only with image data, the spectroscopy component here is all empty. 
    specQtyList = ['None', 'None', 'None', 'None']
    specUnitsList = ['None', 'None', 'None', 'None']
    specVecList = [[int(0)], [int(0)], [int(0)], [int(0)]]
    
    # Need to convert 2D image   data to 1D for writing to h5USID
    imgNp1D = []
    for trace in range(numTraces):
        imgTraceNp = imgNp[trace]
        numPos = imgTraceNp.shape[0] * imgTraceNp.shape[1]
        imgTrace1D = imgTraceNp.reshape(numPos,1)
        imgNp1D.append(imgTrace1D)
        
    # Create the top level group for the STM topography data 
    h5mTopoGroup = usid.hdf_utils.create_indexed_group(h5mfile, 'Topography') # contains the STM data
    
    # Write each of the images (forward up, backward up, etc) to a new group in the h5USID file
    h5mTraceGroupList = []
    for trace in range(numTraces):

        # Create the subgroup for each of the traces. 
        h5mTraceGroup = usid.hdf_utils.create_indexed_group(h5mTopoGroup, traceNameList[trace]) # for the raw STM image
        h5mTraceGroupList.append(h5mTraceGroup)
        # Define the dimensions of the data for writing the h5USID format data.
        posDims = [usid.Dimension(xQty[trace], xUnits[trace], xVec[trace]),
                   usid.Dimension(yQty[trace], yUnits[trace], yVec[trace])]
        
        specDims = usid.Dimension(specQtyList[trace], specUnitsList[trace], specVecList[trace])

        # Get the name of the top level group (will be Topography_000 if created with SPMPy)
        #h5TopoGroupName = list(h5file.keys())[0]

        # Get the h5 group of the top level group. 
        h5TopoGroup = h5file[h5TopoGroupName]
        
        # Write the data to the h5 file
        h5main = usid.hdf_utils.write_main_dataset(h5mTraceGroup,  # parent HDF5 group
                                               imgNp1D[trace],  # 
                                               imgNameList[trace],  # Name of main dataset
                                               mainDataName[trace],  # Physical quantity contained in Main dataset
                                               mainUnits[trace],  # Units for the physical quantity
                                               posDims,  # Position dimensions
                                               specDims,  # Spectroscopic dimensions
                                               dtype=imgNp1D[trace].dtype,  # data type / precision
                                               compression='gzip')
            
    # This command copies all the meta data from the original file to the new file
    sidpy.hdf.hdf_utils.copy_attributes(h5TopoGroup, h5mTopoGroup)
    
    if selectone == None:
        # Copy the meta data for each of the image groups as well (this is the important step)
        for trace in range(numTraces):
            traceGroup = h5file[traceGroupNameList[trace]]
            traceMGroup = h5mfile[traceGroupNameList[trace]]
            sidpy.hdf.hdf_utils.copy_attributes(traceGroup,traceMGroup)
    elif selectone != None:
        # Write the meta data
        imgFileName = h5_get_metadata_parameter(h5FullName,'Original File')[imgSelect]
        MTRXcreationDateString = h5_get_metadata_parameter(h5FullName,'MTRX file creation date string')[imgSelect]
        MTRXcreationDate = h5_get_metadata_parameter(h5FullName,'MTRX file creation date')[imgSelect]
        dataFilePath = h5_get_metadata_parameter(h5FullName,'MTRX File Path')[imgSelect]
        imgNumStr = h5_get_metadata_parameter(h5FullName,'Image Number')[imgSelect]
        traceName = h5_get_metadata_parameter(h5FullName,'Trace Name')[imgSelect]
        imgBias = h5_get_metadata_parameter(h5FullName,'Image Bias')[imgSelect]
        imgCurrent = h5_get_metadata_parameter(h5FullName,'Image Current')[imgSelect]
        imgBiasUnits = h5_get_metadata_parameter(h5FullName,'Bias Units')[imgSelect]
        imgCurrentUnits = h5_get_metadata_parameter(h5FullName,'Current Units')[imgSelect]
        sidpy.hdf_utils.write_simple_attrs(h5mTraceGroup, {'Original File': imgFileName,
                                                 'MTRX file creation date string': MTRXcreationDateString,
                                                 'MTRX file creation date': MTRXcreationDate,
                                                 'MTRX File Path': dataFilePath,
                                                 'Image Number': imgNumStr,
                                                 'Trace Name': traceName,                                         
                                                 'Image Bias': imgBias,
                                                 'Image Current': imgCurrent,
                                                 'Bias Units': imgBiasUnits,
                                                 'Current Units': imgCurrentUnits})

    h5file.close()
    h5mfile.close()    
    return h5mFullName

   
def h5_plot_basic(h5FullName,**kwargs):
    """
    Plot all the traces (forward up, backward up, etc.) contained in a h5USID
    file that has been converted from a MTRX file
    """
    clim = kwargs.get('clim', None)
  #  if clim is not None:
  #      clim = [clim[0] * 1e10, clim[1] * 1e10]
    
    plt.rcParams['savefig.facecolor'] = "1"
    plt.rcParams['figure.figsize'] = 13,13      ## figure size in inches (width, height)
    plt.rcParams['figure.max_open_warning'] = 50
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 20
    
    # Read back in the h5USID file from disk
    h5file = h5py.File(h5FullName, mode='r+')

    # Get the main datasets
    h5main = usid.hdf_utils.get_all_main(h5file) # Finds all the main data sets. 
    numTraces = len(h5main)
    
    # get the 2D image data from the h5USID object
    imgList = []
    for trace in range(numTraces):
        imgList.append(h5main[trace].get_n_dim_form().squeeze())
        
    # Get the auxilliary data
    yVecList, xVecList, yUnitList, xUnitList = h5_get_xyVec(h5FullName)
    
    # Make figure axes (two options depending on whether the down images are present in the MTRX file)
    if numTraces > 1:
        numCols = 2
        numRows = round(numTraces / 2)
    else:
        numCols = 1
        numRows = 1
        
    fig, axes = plt.subplots(ncols=numCols, nrows=numRows)
    
    #theCmap = sidpy.viz.plot_utils.cmap_hot_desaturated()
    theCmap = spiepy.NANOMAP
    
    # Plot the data
    if numTraces > 1:
        for axis, img, yVec, xVec in zip(
                                    axes.flat,
                                    imgList,
                                    yVecList,
                                    xVecList):
            theimg, thecbar = sidpy.viz.plot_utils.plot_map(axis, img, cmap = theCmap,clim=clim,
                                         x_vec=xVec, y_vec=yVec, num_ticks=3)
        _ = fig.tight_layout()
    # Plot the data
    if numTraces == 1:
        theimg, thecbar = sidpy.viz.plot_utils.plot_map(axes, imgList[0], cmap = theCmap,clim=clim,
                                         x_vec=xVecList[0], y_vec=yVecList[0], num_ticks=3)
    #return fig, axes
        
    h5file.close()


def h5_plot_MTRXformat(h5FullName,**kwargs):
    """
    Plot all the traces (forward up, backward up, etc.) contained in a h5USID
    file that has been converted from a MTRX file
    """
    clim = kwargs.get('clim', None)
#    if clim is not None:
#        clim = [clim[0] * 1e10, clim[1] * 1e10]
    
    # Style options for the image plots
    plt.rcParams['savefig.facecolor'] = "1"
    plt.rcParams['figure.figsize'] = 13,13     ## figure size in inches (width, height)
    plt.rcParams['figure.max_open_warning'] = 50
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 14
    
    # Read back in the h5USID file from disk
    h5file = h5py.File(h5FullName, mode='r')

    # Get the main datasets
    h5main = usid.hdf_utils.get_all_main(h5file) # Finds all the main data sets. 
    numTraces = len(h5main)
    
    if numTraces == 1:
        plt.rcParams['figure.figsize'] = 8,8  
        
    # Get the auxilliary data
    yVecList, xVecList, yUnitList, xUnitList = h5_get_xyVec(h5FullName)
    
    for trace in range(numTraces):
        if yUnitList[0] != "m":
            print(trace, "WARNING: displayed Y units may be incorrect. Have assumed loaded data was in m.")
        if xUnitList[0] != "m":
            print(trace, "WARNING: displayed X units may be incorrect. Have assumed loaded data was in m.")
        
    # Change units of the data from SI units for x and y for nice plotting
    maxX = max(xVecList[0])
    xyExponent = 1
    for i in range(3,18,3):
        if maxX * (10 ** i) > 1:
            xyExponent = i
            break
    if xyExponent == 3:
        xyUnit = 'mm'
    elif xyExponent == 6:
        xyUnit = 'um'
    elif xyExponent == 9:
        xyUnit = 'nm'
    elif xyExponent == 12:
        xyUnit = 'pm'
    xVecList = np.asarray(xVecList) * 10 ** xyExponent
    yVecList = np.asarray(yVecList) * 10 ** xyExponent
    xUnits = [xyUnit, xyUnit, xyUnit, xyUnit]
    yUnits = [xyUnit, xyUnit, xyUnit, xyUnit]
    # get the 2D image data from the h5USID object
    imgList = []
    for trace in range(numTraces):
        imgList.append(h5main[trace].get_n_dim_form().squeeze())
    
    # Make figure axes (two options depending on whether the down images are present in the MTRX file)
    if numTraces > 1:
        numCols = 2
        numRows = round(numTraces / 2)
    else:
        numCols = 1
        numRows = 1
        
    # Change units of the main data data from SI units for nice plotting
    maxData = np.amax(imgList[0])
    dataExponent = 0
    dataUnit=""
    for i in range(3,18,3):
        if maxData * (10 ** i) > 1:
            dataExponent = i
            break
    if dataExponent == 3:
        dataUnit = 'mm'
    elif dataExponent == 6:
        dataUnit = 'um'
    elif dataExponent == 9:
        dataUnit = 'nm'
    elif dataExponent == 12:
        dataUnit = 'pm'
    mainUnits = [dataUnit, dataUnit, dataUnit, dataUnit]
    
    imgList = np.asarray(imgList)
    imgList = imgList * 10 ** dataExponent
        
    # Get the name of the top level group (will be Topography_000 if created with SPMPy)
    h5TopoGroupName = list(h5file.keys())[0]
    
    # Get the h5 group of the top level group. 
    h5TopoGroup = h5file[h5TopoGroupName]
    
    # read metadata from the h5USID file
    traceGroupNameList, imgNameList, dateList, biasList, currentList, biasUnitsList, currentUnitsLists = \
    h5_get_metadata_parameters_for_image_title(h5FullName)
    # check that metadata was read
    if len(imgNameList) < numTraces:
        for trace in range(numTraces):
            imgNameList.append('')
    if len(dateList) < numTraces:
        for trace in range(numTraces):
            dateList.append('Unknown date')
    if len(biasList) < numTraces:
        for trace in range(numTraces):
            biasList.append('??')
    if len(currentList) < numTraces:
        for trace in range(numTraces):
            currentList.append('??')
    if len(biasUnitsList) < numTraces:
        for trace in range(numTraces):
            biasUnitsList.append('V')
    if len(currentUnitsLists) < numTraces:
        for trace in range(numTraces):
            currentUnitsLists.append('nA')
       
    # create figure subpanel titles
    titleList  = []
    for trace in range(numTraces):
        titleList.append(dateList[trace]+"\n"+\
        imgNameList[trace]+" ("+\
        str(biasList[trace])+" "+biasUnitsList[trace]+", "+str(currentList[trace])+" "+currentUnitsLists[trace]+")")
        
    # for the purpose of displaying the images, we reorder them
    if numTraces == 4:
        imgListReordered = [imgList[3],imgList[1],imgList[2],imgList[0]]
        xVecListReordered = [xVecList[3],xVecList[1],xVecList[2],xVecList[0]]   
        yVecListReordered = [yVecList[3],yVecList[1],yVecList[2],yVecList[0]]  
        titleListReordered = [titleList[3], titleList[1], titleList[2], titleList[0]]  
    elif numTraces == 2:
        imgListReordered = [imgList[1],imgList[0]]
        xVecListReordered = [xVecList[1],xVecList[0]] 
        yVecListReordered = [yVecList[1],yVecList[0]]  
        titleListReordered = [titleList[1], titleList[0]]
    else:
        imgListReordered = imgList
        xVecListReordered = xVecList
        yVecListReordered = yVecList  
        titleListReordered = titleList
        
    # Plot the data
    if numTraces > 1:
        fig, axes = plt.subplots(ncols=numCols, nrows=numRows)
        for axis, img, yVec, xVec, title in zip(
                                    axes.flat, 
                                    imgListReordered,
                                    yVecListReordered,
                                    xVecListReordered,
                                    titleListReordered):
            theimg, thecbar = sidpy.viz.plot_utils.plot_map(axis, img,
                                         cmap = spiepy.NANOMAP,clim=clim,
                                         x_vec=xVec, y_vec=yVec, num_ticks=3)
            _ = thecbar.set_label('Tip height ('+mainUnits[trace]+')')
            _ = axis.set_title(title)
            _ = axis.set_xlabel('Distance x ('+xUnits[trace]+')')
            _ = axis.set_ylabel('Distance y ('+yUnits[trace]+')')
            _ = fig.tight_layout()
    elif numTraces == 1:
        plt.rcParams['figure.figsize'] = 11,11
        fig, axes = plt.subplots(ncols=numCols, nrows=numRows)
        img = imgListReordered[0]
        yVec = yVecListReordered[0]
        xVec = xVecListReordered[0]
        title = titleListReordered[0]
        theimg, thecbar = sidpy.viz.plot_utils.plot_map(axes, img,
                                     cmap = spiepy.NANOMAP,clim=clim,
                                     x_vec=xVec, y_vec=yVec, num_ticks=3)
        _ = thecbar.set_label('Tip height ('+mainUnits[trace]+')')
        _ = axes.set_title(title)
        _ = axes.set_xlabel('Distance x ('+xUnits[trace]+')')
        _ = axes.set_ylabel('Distance y ('+yUnits[trace]+')')
       # _ = fig.tight_layout()
    h5file.close()
    
    

def h5_show_MTRXinfo(h5FullName,**kwargs):
    """
    think this function is a little redundant now, but leaving it in case
    it is useful for reference later. 
    """
    display = kwargs.get('display', 'yes')
    
    # Read back in the h5USID file from disk
    h5file = h5py.File(h5FullName, mode='r')

    # Get the main datasets
    h5main = usid.hdf_utils.get_all_main(h5file) # Finds all the main data sets. 
    numTraces = len(h5main)
    if display == 'yes':
        print('============================== START =======================================')
        print()
        print('----------------------------------------------------------------------------')
        print("Opened file:")
        print('----------------------------------------------------------------------------')
        print(h5FullName)
    
    # Get the name of the top level group (will be Topography_000 if created with SPMPy)
    h5TopoGroupName = list(h5file.keys())[0]
        
    # Get the h5 group of the top level group. 
    h5TopoGroup = h5file[h5TopoGroupName]

    if display == 'yes':
        print()
        print('----------------------------------------------------------------------------')
        print('Top level group:')
        print('----------------------------------------------------------------------------')
        print(h5TopoGroupName)

    # Get all the main data sets in this group. 
    mainDataSets = usid.hdf_utils.get_all_main(h5TopoGroup)
    
    traceFullNameList = []
    if display == 'yes':
        print()
        print('----------------------------------------------------------------------------')
        print('Trace names:')
        print('----------------------------------------------------------------------------')
        
    for trace in range(numTraces):
        traceFullNameList.append(mainDataSets[trace].name)
        if display == 'yes':
            print(traceFullNameList[trace])    
    
    traceGroupNameList = []
    if display == 'yes':
        print()
        print('----------------------------------------------------------------------------')
        print('Trace group names:')
        print('----------------------------------------------------------------------------')
    for trace in range(numTraces):
        imgGroupName = pathlib.PurePath(traceFullNameList[trace]).parent
        traceGroupNameList.append(str(imgGroupName))
        if display == 'yes':
            print(traceGroupNameList[trace])
 
    imgNameList = []
    if display == 'yes':
        print()
        print('----------------------------------------------------------------------------')
        print('Image names:')
        print('----------------------------------------------------------------------------')
    for trace in range(numTraces):
        imgName = pathlib.PurePath(traceFullNameList[trace]).name
        imgNameList.append(str(imgName))
        if display == 'yes':
            print(imgNameList[trace])
       
    if display == 'yes':
        print('=============================== END ========================================')
        
    h5file.close()
    

def get_2Dnumpy(h5FullName):
    
    # Read back in the h5USID file from disk
    h5file = h5py.File(h5FullName, mode='r')

    # Get the main datasets
    h5main = usid.hdf_utils.get_all_main(h5file) # Finds all the main data sets. 
    numTraces = len(h5main)
    
    # get the 2D image data from the h5USID object
    imgList = []
    for trace in range(numTraces):
        imgList.append(h5main[trace].get_n_dim_form().squeeze())
    
    return imgList
    
    

def h5USID_info(h5FullName):
    """
    Show all the information about a h5USID file
    """
    # Load the h5USID file
    h5file = h5py.File(h5FullName, mode='r')

    print('---------------------------------')
    print('Tree structure for: ', h5FullName.split('/')[-1])
    print('---------------------------------')
    # sidpy routine to display tree structure of h5USID file
    sidpy.hdf_utils.print_tree(h5file, rel_paths=True)

    # Get the main datasets
    mainDataSets = usid.hdf_utils.get_all_main(h5file) # Finds all the main data sets. 
    numTraces = len(mainDataSets)

    # Get the name of the top level group
    h5TopoGroupName = list(h5file.keys())[0]

    #Assign the top group
    h5TopoGroup = h5file[h5TopoGroupName]

    print('---------------------------------')
    print('Meta data in group (key, val): ',h5TopoGroupName)
    print('---------------------------------')
    template = "{0:20}{1:60}"
    for key, val in sidpy.hdf_utils.get_attributes(h5TopoGroup).items():
            print(template.format(key, val))

    traceFullNameList = []  
    print('---------------------------------')
    print('Items in traceFullNameList')
    print('---------------------------------')
    for trace in range(numTraces):
        traceFullNameList.append(mainDataSets[trace].name)
        print(traceFullNameList[trace])

    # Separate group and image names
    traceGroupNameList = []
    imgNameList = []
    for trace in range(numTraces):
        traceFullName = traceFullNameList[trace]
        imgName = traceFullName.split('/')[-1]    
        imgNameLen=len(imgName)
        traceGroupName = traceFullName[:-imgNameLen]
        traceGroupNameList.append(traceGroupName)
        imgNameList.append(imgName)

    print('---------------------------------')
    print('Items in traceGroupNameList')
    print('---------------------------------')
    for trace in range(numTraces):
        print(traceGroupNameList[trace])

    print('---------------------------------')
    print('Items in imgNameList')
    print('---------------------------------')
    for trace in range(numTraces):
        print(imgNameList[trace])

        #    imgGroupName = pathlib.PurePath(traceFullNameList[trace]).parent
    #    print('imgGroupName', imgGroupName)
    ##    imgName = traceFullNameList[trace].split('/')[-1]
    #    traceGroupNameList.append(str(imgGroupName))
    #    traceNameList.append(str(imgName))
    #    traceGroup = h5file[traceGroupNameList[trace]]

    #    print('---------------------------------')
    #    print(traceGroupNameList[trace])
    #    print(traceNameList[trace])
    #    print('---------------------------------')
        # Print to screen the meta data of the file - the top level group.
       # for key, val in sidpy.hdf_utils.get_attributes(traceGroup).items():
       #     print('{} : {}'.format(key, val))

    for trace in range(numTraces):
        traceGroupName = traceGroupNameList[trace]
        print('---------------------------------')
        print('Meta data in group (key, val): ',traceGroupName)
        print('---------------------------------')

        traceGroup = h5file[traceGroupName]
        template = "{0:25}: {1:100}"
        for key, val in sidpy.hdf_utils.get_attributes(traceGroup).items():
                print(template.format(key, str(val)))


    # return traceGroupNameList, imgNameList
    # Close the file.
    h5file.close()



# Borrowed from spiepy - trying to understand how to alter color maps;
def nanomap():
    begin_color = '#000000'
    mid_color = '#ff8000'
    end_color = '#ffffff'
    c_list = [begin_color, mid_color, end_color]
    return mcolors.LinearSegmentedColormap.from_list('nanomap', c_list)


















