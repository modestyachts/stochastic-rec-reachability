""" Configuration values"""
import os

# specify the location of the data, experiments and model folders
folder , _ = os.path.split(__file__)

# contains folders with historical recommendation data
DATAPATH = os.path.normpath(os.path.join(folder, '../data'))
# containts folders with experimental runs
EXPPATH = os.path.normpath(os.path.join(folder, '../exp'))
# contains trained models and optimal training parameters
MODELPATH = os.path.normpath(os.path.join(folder, '../models'))
