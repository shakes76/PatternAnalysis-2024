'''
[desc]
script to setup the project directories
and validate that the data is supplied

@author Jamie Westerhout
@project Stable Diffusion
@date 2024
'''
import os
import zipfile

def create_paths():
    '''
        create paths need for training to work
        models folder for it to be able to out put the models to
        and output path to output logs to

        Returns:
            tuple of (model path, out path)
    '''

    #setup model path
    try:
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
        os.mkdir(model_path)
        print("model path created")
    except FileExistsError:
        print("model path already exists")
    except:
        print("failed to create model path")
    
    #setup output path
    try:
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))
        os.mkdir(output_path)
        print("output path created")
    except FileExistsError:
        print("output path already exists")
    except:
        print("failed to create output path")

    return (model_path, output_path)

def decompress_data(archive_path, output_path):
    '''
        decompressed the file at archive path
        the puts the output in output path

        returns:
            True if succsessful
            False if it fails
    '''
    print("decompressing.....")
    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_path) #decompress
            print("decompressed")
            return True
    except:
        print("decompression failed")
        return False

def decompressable_check(data_path):
    '''
        checks to see if the ADNI.zip archive exsits in the data path
        if it does then decompress it and return True otherwise return false

        Returns:
            True if ADNI.zip exists and decompressed completed
            False if file not found or fails to decompress
    '''
    adni_archive_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI.zip'))
    if os.path.isfile(adni_archive_data_path):
        print("data archive found")
        output = decompress_data(adni_archive_data_path, data_path)
        if not output:
            return False
        print("data set extracted, data should be okay now")
        return True
    return False

def verify_data_set():
    '''
        verfies that dataset is avaliable 
        if its not try and fix and the ask the user to supply missing data
        then corrects the issues if the user supplies the correct files
    '''
    adni_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI'))
    if os.path.isdir(adni_data_path):
        print("data set root found, data should be okay")
    else:
        print("data not found")
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
        if os.path.isdir(data_path):
            print("data folder found")
            if not decompressable_check(data_path):
                while True:
                    input("Please place ADNI.zip archive in the data folder, press enter once complete")
                    if decompressable_check(data_path):
                        break
        else:
            print("making data path")
            os.mkdir(data_path)
            while True:
                input("Please place ADNI.zip archive in the data folder, press enter once complete")
                if decompressable_check(data_path):
                    break
            
if __name__ == '__main__':
    create_paths()
    verify_data_set()
                