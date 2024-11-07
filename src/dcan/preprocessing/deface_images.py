import shutil
from pydeface.utils import deface_image
import os


#This function goes through all of the files and subdirectories in the file path specified by the user
#and appends all filepaths to a list. The function then returns this list
def walk_files(src_filepath = "."):
    filepath_list = []
   
    #This for loop uses the os.walk() function to walk through the files and directories
    #and records the filepaths of the files to a list
    for root, dirs, files in os.walk(src_filepath):
        
        #iterate through the files currently obtained by os.walk() and
        #create the filepath string for that file and add it to the filepath_list list
        for file in files:
            #Checks to see if the root is '.' and changes it to the correct current
            #working directory by calling os.getcwd(). Otherwise root_path will just be the root variable value.
            if root == '.':
                root_path = os.getcwd() + "/"
            else:
                root_path = root
            
            #This if statement checks to see if an extra '/' character is needed to append 
            #to the filepath or not
            if (root_path != src_filepath) and (root != '.'):
                filepath = root_path + "/" + file
            else:
                filepath = root_path + file
            
            #Appends filepath to filepath_list if filepath does not currently exist in filepath_list
            if filepath not in filepath_list:
                filepath_list.append(filepath)
                
    #Return filepath_list        
    return filepath_list


def deface_images(input_folder, output_folder):
    """Run the deface utility for images."""
    shutil.copytree(input_folder, output_folder)
    files = walk_files(output_folder)
    # deface_image()
