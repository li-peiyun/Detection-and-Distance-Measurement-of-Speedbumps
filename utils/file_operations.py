import os

def cleanFolder(folder):
    # delete the former photos
    del_list = os.listdir(folder)
    for f in del_list:
        file_path = os.path.join(folder, f)
        if os.path.isfile(file_path):
            os.remove(file_path)


def writeIntriToFile(output_filename, cameraMatrix, distCoeffs):
    with open(output_filename, mode='w', encoding='utf-8') as output_file:
        output_file.write("import numpy as np\n")

        output_file.write("cameraMatrix = np.float32(" + str(cameraMatrix.tolist()) + ')')

        output_file.write("\ndistCoeff = np.float32(" + str(distCoeffs.tolist()) + ')')

    print("intrinsics saved to", output_filename)

