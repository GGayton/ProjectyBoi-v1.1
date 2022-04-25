import os

def recursiveFileFinder(directory, fileType):
    
    dirList = os.listdir(directory)
    
    print(r"[0] - \\")
    
    for i in range(0, len(dirList)):
        print("[{}] - ".format(i+1) + dirList[i])
        
    choice = int(input("Choose file:"))
    
    if choice == 0:
        
        if directory[-2:] == "\\":directory = directory[:-2]
        
        cap = directory[::-1].find("\\") + 1
        
        print(cap, directory)
        
        directory = directory[:-cap]
        
        return recursiveFileFinder(directory, fileType)
    
    elif dirList[choice-1][-len(fileType):] == fileType:
        
        return directory + "\\" + dirList[choice-1]
    
    else:
        
        return recursiveFileFinder(directory + "\\" + dirList[choice-1], fileType)


