# roadmap_parser.py
# Author: Nic Trieu
# An assortment of methods for parsing and analyzing pretty-printed songs.

# EXAMPLE USAGE:

#   Generating roadmaps from a directory of pretty-printed songs:
# roadmaps = generateRoadmaps(directory, tempDirectory, doTranspose=True)

#   Generating a dictionary of bricks and chords:
# bricks,titles,brickUsages,patterns,chordslist,keyUsages=generateBricks(roadmaps,printVerbose=True)

#   Plotting the key usages:
# keyPlot(keyUsages,"Key usages (Brick label criteria: name,key)",
#     fname=r"C:\Users\Nic\Documents\Improvisor\media\keyUsage_nameKey.png",
#        show=True)

#   Plotting the brick usages:
# normPlot(brickUsages,"Brick usages (Brick label criteria: name,key)",
#     fname=r"C:\Users\Nic\Documents\Improvisor\brick_usages\brickUsage_namekey.png",
#        show=True)

#   Getting and plotting the transition probabilities between bricks:
# brick_keys,probmat=generateProbmats(roadmaps,titles)
# plotProbmat(len(brick_keys),probmat)

#   Printing the brick and key transition probabilities for a specific brick:
# printBrickTrans(brick_keys,probmat,('Starlight-Cadence','C'))

#   Printing the brick and key transition probabilities for all the bricks:
# printNextBricks(brick_keys,probmat,len(brick_keys),printbricks=True,printkeys=True,sortProb=True)



import os
import operator
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import copy
from operator import itemgetter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

NAME_INDEX = 0
VARIANT_INDEX = 1
TYPE_INDEX = 2
KEY_INDEX = 3
MODE_INDEX = 4
DUR_INDEX = 5
OVERLAP_INDEX = 6
CHORDKEY_INDEX = 0
CHORDDUR_INDEX = 1
KEY_LABELS = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

def getRoadmapString(songString):
    """
    Cut out the roadmap section of a pretty-printed song.

        songString: String representing the pretty-printed contents. 
    """
    songString = songString.replace('(', " ( ")
    songString = songString.replace(')', " ) ")
    tokens = songString.split()    
    index = 0
    parensToClose = 1
    numTokens = len(tokens)
    startIndex = 0
    while index < numTokens:
        if tokens[index] == "roadmap":
            startIndex = index
            break
        index+=1
    while index < numTokens and parensToClose > 0:
        if tokens[index] == "(":
            parensToClose += 1
            index+=1
            continue
        if tokens[index] == ")":
            parensToClose -= 1
            index+=1
            continue
        index+=1
    endIndex = index
    roadmapTokens = tokens[startIndex:endIndex]
    roadmapString = reduce(lambda x,y: str(x)+"\n"+str(y), roadmapTokens, "")
    return roadmapString.strip()

def parseMap(file):
    """
    Parse the roadmap section of a pretty-printed song.

        file: Mod file containing a formatted roadmap from a pretty-printed song.
    """
    ident = file.readline().strip()
    mapArray = [ident,[]]
    while True:
        nt = file.readline().strip()
        if nt == ")":
            break
        elif nt == "(":
            subMap = parseMap(file)
            mapArray[1].append(subMap)
        else:
            mapArray[1].append(nt)
    return mapArray

def transposeBrick(nest,keyDif,verbose=False):
    """
    Transposes a brick up by keyDif half-steps.
    """
    if not isinstance(nest, (list,tuple)):
        return nest
    if len(nest) < 2:
        return nest
    ident = nest[0]
    if ident == "chord":
        chordkey = nest[1][CHORDKEY_INDEX]
        if len(chordkey)>1 and chordkey[0:2] in KEY_LABELS:
            keyval = KEY_LABELS.index(chordkey[0:2])-keyDif
            newKey = KEY_LABELS[keyval % len(KEY_LABELS)]
            nest[1][CHORDKEY_INDEX] = newKey+chordkey[2:]
        elif chordkey[0] in KEY_LABELS:
            keyval = KEY_LABELS.index(chordkey[0])-keyDif
            newKey = KEY_LABELS[keyval % len(KEY_LABELS)]
            nest[1][CHORDKEY_INDEX] = newKey+chordkey[1:]
    else:
        if ident == "brick":
            brickkey = nest[1][KEY_INDEX][1][0]
            if len(brickkey) > 1 and brickkey[0:2] in KEY_LABELS:
                keyval = KEY_LABELS.index(brickkey[0:2])-keyDif
                newKey = KEY_LABELS[keyval % len(KEY_LABELS)]
                nest[1][KEY_INDEX][1][0] = newKey+brickkey[2:]
            elif brickkey[0] in KEY_LABELS:
                keyval = KEY_LABELS.index(brickkey[0])-keyDif
                newKey = KEY_LABELS[keyval % len(KEY_LABELS)]
                nest[1][KEY_INDEX][1][0] = newKey+brickkey[1:]
        for i in range(len(nest[1])):
            val = nest[1][i]
            nest[1][i] = transposeBrick(val,keyDif)
    return nest

def transposeRoadmap(blocklist,keyDif):
    """
    Transposes a roadmap up by keyDif half-steps.
    """
    for i in range(len(blocklist)):
        block = blocklist[i]
        blocklist[i] = transposeBrick(block,keyDif)
    return blocklist
            


# In[3]:

def generateRoadmaps(path,mod_path,doTranspose=False):
    """
    Generate a list of parsed roadmaps.

        path: Filepath to a directory of pretty-printed roadmaps.
        mod_path: Filepath to an empty directory, used for intermediate parsing steps.
        doTranspose: If true, then transposes all songs to C-major, else does not transpose.
    """
    #path = r"C:\Users\Nic\Documents\Improvisor\insightsRoadmaps"
    #mod_path = r"C:\Users\Nic\Documents\Improvisor\insightsRoadmaps_mod"
    roadmaps = []
    for filename in os.listdir(path):
        songString = ""
        f = open(path+"/"+filename,'r')
        modf = open(mod_path+"/"+filename,'w')
        for line in f:
            songString+=line.strip()
        songString = getRoadmapString(songString)
        modf.write(songString)
        f.close()
        modf.close()
    for filename in os.listdir(mod_path):
        modf = open(mod_path+"/"+filename,'r')
        roadmap = parseMap(modf)
        if doTranspose:
            bricklist = roadmap[1][0][1]
            firstThing = bricklist[0]
            if firstThing[0] == 'chord':
                firstChord = firstThing[1]
                keyDif = KEY_LABELS.index(firstChord[0][0])
            else:
                firstBrick = bricklist[0][1]
                keyDif = KEY_LABELS.index(firstBrick[KEY_INDEX][1][0])
            if keyDif != 0:
                roadmap[1][0][1]=transposeRoadmap(bricklist,keyDif)
        roadmaps.append((filename,roadmap))
        #print(roadmap)
        f.close()
    return roadmaps

# In[4]:

def printBrickNames(roadmaps,index):
    """
    Prints the bricks of a particular roadmap.

        roadmaps: List of roadmaps.
        index: Integer denoting which roadmap to parse.
    """
    target = roadmaps[index]
    print("Song: " + target[0])
    roadmap = target[1]
    blocks = roadmap[1][0]
    for block in blocks[1]:
        ident = block[0]
        vals = block[1]
        if ident == "brick":
            name = ""
            for nameParts in vals[NAME_INDEX][1]:
                name+=nameParts
            print("Brick: " + name)
        elif ident == "chord":
            chordkey = vals[CHORDKEY_INDEX]
            chorddur = vals[CHORDDUR_INDEX]
            print("Chord (key, dur): " + chordkey + ", " + chorddur)
            
def parseRoadmap(roadmap,brickList=[],brickDict={},multChords = False, surfaceList = False,
    includeName=True,includeKey=False,includeVar=False, surface=False,  
    deep=False, includeDur=False, verbose=False, verboseDupe=False):
    """
    Parse the roadmap into bricks, creating a label denoting each brick.
    Stores the brick labels in a list if only the brick descriptions are desired, 
    or else stores the chords for each brick in a dictionary using the brick labels.

    Brick labels have up to 3 main features (though others could be used):
    1. name
    2. key
    3. variant

    Chord labels have up to 2 main features:
    1. Key
    2. Duration

    The dictionary uses the brick labels as the keys, and stores the following list:
    [numberOfBrickUses, numberOfChordPatterns, listOfPossibleChordPatterns]

    Each chord pattern represents a different set of chord labels found for the same brick label.

    Inputs:
        roadmap: The roadmap to parse.
        brickList: List storing the brick labels.
        brickDict: Dictionary storing the chords for each brick label.
        multChords: True if the dictionary is desired.
        surfaceList: True if the list is desired.
        includeName: If true, include the name in the brick label.
        includeKey: If true, include the key in the brick label.
        includeVar: If true, include the variant in the brick label.
        surface: If true, only parse the surface-level bricks, do not parse sub-bricks.
        deep: If true, parse all bricks including the sub-bricks.
        includeDur: If true, include the chord duration in the chord label.
    """
    target = roadmap
    song_name = target[0]
    if verbose:
        print("Song: " + song_name)
    roadmap = target[1]
    blocks = roadmap[1][0]
    for block in blocks[1]:
        ident = block[0]
        vals = block[1]
        if ident == "brick":
            name = ""
            for nameParts in vals[NAME_INDEX][1]:
                name+=nameParts
            key = ""
            for keyParts in vals[KEY_INDEX][1]:
                key+=keyParts
            variant = ""
            for varParts in vals[VARIANT_INDEX][1]:
                variant += varParts
            dict_key = []
            if includeName:
                dict_key.append(name)
            if includeKey:
                dict_key.append(key)
            if includeVar:
                dict_key.append(variant)
            dict_key = tuple(dict_key)
            if surface:
                if verbose:
                    print("Brick: " + name + ". Key: " + key + ". Variant: " + variant + ".")
                if dict_key not in brickDict:
                    chords = tuple(chordFinder(block,durOn=includeDur, verbose=verbose))
                    brickDict[dict_key] = (song_name,chords)
                else:
                    chords = tuple(chordFinder(block,durOn=includeDur,verbose=verbose))
                    if chords == brickDict[dict_key][1]:
                        #print("Same dupe")
                        pass
                    elif verboseDupe:
                        print("Diff dupe. Song: " + song_name + ". Brick: " + name + ". Key: "+ key + ". Variant: " + variant + ".")
                        print(chords)
                        print("Orig song: " + brickDict[dict_key][0])
                        print(brickDict[dict_key][1])
            elif surfaceList:
                if verbose:
                    print("Brick: " + name + ". Key: " + key + ". Variant: " + variant + ".")
                brickList.append(dict_key)
            elif multChords:
                if verbose:
                    print("Brick: " + name + ". Key: " + key + ". Variant: " + variant + ".")
                if dict_key not in brickDict:
                    brickDict[dict_key] = [0,0,[]]
                chords = tuple(chordFinder(block,durOn=includeDur,verbose=verbose))
                if chords not in brickDict[dict_key][2]:
                    brickDict[dict_key][2].append(chords)
                    brickDict[dict_key][1]+=1
                brickDict[dict_key][0] += 1
                
            elif deep:
                chordFinder_Brick(block,name)
            else:
                chordFinder(block,verbose=verbose)
        elif ident == "chord":
            chordkey = vals[CHORDKEY_INDEX]
            chorddur = vals[CHORDDUR_INDEX]
            if verbose:
                print("Chord (key, dur): " + chordkey + ", " + chorddur)
    if surfaceList:
        return brickList
    elif multChords:
        return brickDict

def chordFinder(nest, durOn=False, verbose=False):
    """
    Helper method to find all chords in a brick nest.
    Returns all chords found.
    """
    if not isinstance(nest, (list,tuple)):
        return []
    if len(nest) < 2:
        return []
    ident = nest[0]
    if ident == "chord":
        if verbose:
            chordkey = nest[1][CHORDKEY_INDEX]
            chorddur = nest[1][CHORDDUR_INDEX]
            print("Chord (key, dur): " + chordkey + ", " + chorddur)
        return [tuple(nest[1])] if durOn else [nest[1][CHORDKEY_INDEX]]
    else:
        chords = []
        for val in nest[1]:
            chords+=chordFinder(val,durOn,verbose)
        return chords

def chordFinder_Brick(nest,name,verbose=False):
    """
    Prints out the chords associated with each brick in a nest of bricks.

        name: The top-level brick name of the nest.
    """
    if not isinstance(nest, (list,tuple)):
        return
    if len(nest) < 2:
        return
    ident = nest[0]
    if ident == "chord":
        chordkey = nest[1][CHORDKEY_INDEX]
        chorddur = nest[1][CHORDDUR_INDEX]
        if verbose:
            print("Chord ("+name+"): " + chordkey + ", " + chorddur)
    else:
        if ident == "brick":
            name = ""
            for nameParts in vals[NAME_INDEX][1]:
                name+=nameParts
            if verbose:
                print("Brick: " + name)
        for val in nest[1]:
            chordFinder_Brick(val,name)


# In[5]:

def sortAlphaTitle(multbricks):
    """
    Returns a list of bricks sorted alphabetically by their name.

        multbricks: A dictionary of bricks labels corresponding to chords.
    """
    return sorted(multbricks.items(), key=operator.itemgetter(0))

def sortNumUses(multbricks):
    """
    Returns a list of bricks sorted by their number of usages.

        multbricks: A dictionary of bricks labels corresponding to chords.
    """
    sortedBricks = sorted(multbricks.items(),key=operator.itemgetter(1))
    sortedBricks.reverse()
    return sortedBricks

def countNumUses_eq(brickList,targetCount):
    """
    Count the number of bricks used exactly targetCount times.

        brickList: A list of bricks.
    """
    numBricks = 0
    for brick in sortedmultbricks:
        numUses = brick[1][0]
        if numUses == targetCount:
            numBricks += 1
    return numBricks

def countNumUses_leq(multbricks,targetCount):
    """
    Count the number of bricks used at most targetCount times.

        brickList: A list of bricks.
    """
    numBricks = 0
    for brick in multbricks:
        numUses = brick[1][0]
        if numUses <= targetCount:
            numBricks += 1
    return numBricks


# In[6]:

def printInfo(bricks,sortFn=sortNumUses, usesToCount=3,verbose=False,printChords=False):
    """
    Prints out general information about a brick dictionary.
    Sorts the bricks according to sortFn, then returns several lists in that order:
    1. The titles of the bricks.
    2. The number of usages for each brick.
    3. The number of patterns for each brick.
    4. The list of chords for each brick.
    5. A list of the usages of each key over all the bricks.

        bricks: A dictionary of bricks.
    """
    sortedbricks = sortFn(bricks)
    numFewUseBricks = countNumUses_leq(sortedbricks,usesToCount)
    if verbose:
        print("Total number of bricks: " + str(len(sortedbricks)))
        print("Number of bricks with "+str(usesToCount)+" or fewer uses: " + str(numFewUseBricks))
        print()
    titles = []
    uses = []
    patternCounts = []
    chordslist = []
    keyUsages = [0]*len(KEY_LABELS)
    for brick in sortedbricks:
        title = brick[0]
        titles.append(title)
        for i in range(len(KEY_LABELS)):
            if KEY_LABELS[i] in title:
                keyUsages[i]+=1
        (numUses,numPatterns, chords) = brick[1]
        uses.append(numUses)
        patternCounts.append(numPatterns)
        chordslist.append(chords)
        if verbose:
            print(title)
            print("Num uses: " + str(numUses))
            print("Num chord patterns: " + str(numPatterns))
            if printChords:
                for chordset in chords:
                    print(chordset)
            print()
    return titles,uses,patternCounts,chordslist,keyUsages

def sortChords(bricks):
    """
    Sort the chords of each brick by alphabetical order.
        bricks: A dictionary of bricks.
    """
    for brick in bricks:
        chords = bricks[brick][2]
        bricks[brick][2] = sorted(chords)

def getUses(bricks):
    """
    Get the usages of each brick.
        bricks: A dictionary of bricks.
    """
    uses = []
    for brick in bricks:
        uses.append(bricks[brick][0])
    return uses


# In[7]:

def generateBricks(roadmaps,printVerbose=False):
    """
    Given a list of roadmaps, parse each roadmap into bricks.
    Return a dictionary of bricks, and several lists detailing the bricks.
    """
    bricks = {}
    for roadmap in roadmaps:
        parseRoadmap(roadmap,brickDict=bricks,multChords=True,surface=False,includeVar=False,includeDur=False,verbose=False)
    sortChords(bricks)
    titles,uses,patterns,chordslist,keyUsages=printInfo(bricks,sortFn = sortNumUses,usesToCount=3,printChords=False,verbose=printVerbose)
    return bricks,titles,uses,patterns,chordslist,keyUsages


# In[8]:

def keyPlot(keyUsages,title,fname="dummy",show=True):
    """
    Plot the key usages of all the bricks.
    X-axis: Keys (i.e. C, Db, etc.)
    Y-axis: Number of times a key is used.
    """
    plt.close('all')
    plt.bar(range(len(KEY_LABELS)),keyUsages,0.5,color="blue")
    plt.xticks(range(len(KEY_LABELS)),KEY_LABELS)
    plt.title(title)
    plt.xlabel("Key")
    plt.ylabel("Num bricks")
    if show:
        plt.show()
    else:
        plt.savefig(fname)

def normPlot(brickUsages,title,fname="dummy",show=True):
    """
    Plot the brick usages.
    X-axis: Number of usages.
    Y-axis: Number of bricks that have X many usages.
    """
    plt.close('all')
    plt.hist(brickUsages,100)
    plt.title(title)
    plt.xlabel("Num uses")
    plt.ylabel("Num bricks")
    if show:
        plt.show()
    else:
        plt.savefig(fname)


# In[9]:

def generateProbmats(roadmaps,titles):
    """
    Parse the roadmaps into a list of brick labels.
    Return the transition matrix of the bricks.

        roadmaps: List of roadmaps.
        titles: List of brick labels to use. #TODO Should just recompute this, not pass it in.
    """
    brick_keys = copy.deepcopy(titles)
    #brick_keys.reverse()
    n = len(brick_keys)
    probmat = np.zeros((n,n))
    bricks2 = []
    for roadmap in roadmaps:
        bricklist = parseRoadmap(roadmap,brickList=bricks2,surfaceList=True,
            includeVar=False,includeDur=False,verbose=False,verboseDupe=False)
        for i in range(len(bricklist)-1):
            brick = bricklist[i]
            nextBrick = bricklist[i+1]
            currIndex = brick_keys.index(brick)
            nextIndex = brick_keys.index(nextBrick)
            probmat[currIndex,nextIndex] += 1
    sums = [sum(row) for row in probmat]
    for i in range(n):
        for j in range(n):
            probmat[i,j]/=sums[i]
    return brick_keys, probmat


# In[10]:

def plotProbmat(n,probmat,showPlot=True,savePlot=False):
    """
    Plot the 1st-order transition probabilities of the bricks.
    """
    fig = plt.figure()
    plt.clf()
    ax = fig.gca(projection='3d')
    surface_xrange = range(n)
    surface_yrange = range(n)
    X, Y = np.meshgrid(surface_xrange,surface_yrange)
    surf = ax.plot_surface(X,Y,probmat,cmap=cm.YlOrRd)
    if showPlot:
        plt.show()
    elif savePlot:
        plt.savefig(r"C:\Users\Nic\Documents\Improvisor\media\probmat.png")

def printBrickTrans(brick_keys,probmat,brickLabel):
    """
    Print the transition probabilities for a specific brick.

        brickLabel: A tuple representing the brick label. Must match brick_keys.
    """
    i = brick_keys.index(brickLabel)
    brick = brick_keys[i]
    print("Brick: "+str(brick)+".")
    row = probmat[i]
    nonzerolist = []
    for j in range(len(row)):
        if row[j] != 0:
            nonzerolist.append((row[j],brick_keys[j]))
    nonzerolist.sort(key=itemgetter(0),reverse=True)
    if printbricks:
        for nextBrick in nonzerolist:
            print("  Next: " + str(nextBrick[1]) + ". Prob: " + str(round(nextBrick[0],2)) + ".")
    bkeylist = [[i,0] for i in range(len(KEY_LABELS))]
    for nextBrick in nonzerolist:
        bkey = nextBrick[1]
        if len(bkey) != 2:
            printkeys = False
            break
        key = bkey[1]
        index = KEY_LABELS.index(key)
        bkeylist[index][1] += nextBrick[0]
    if printkeys:
        if sortProb:
            bkeylist.sort(key=itemgetter(1),reverse=True)
        for bkey in bkeylist:
            keyindex = bkey[0]
            keyprob = bkey[1]
            if keyprob != 0:
                print("  Next key: " + KEY_LABELS[keyindex] + ". Prob: " + str(round(keyprob,2)))

def printNextBrick(brick_keys,probmat, numBricksToPrint):
    """
    Print out the most likely next brick for each of the bricks.
    """
    for i in range(numBricksToPrint):
        #i = -i
        brick = brick_keys[i]
        row = probmat[i]
        mlnext_index = 0
        mlnext_prob = row[0]
        for j in range(len(row)):
            if row[j] > mlnext_prob:
                mlnext_prob = row[j]
                mlnext_index = j
        mlnext_brick = brick_keys[mlnext_index]
        print("Brick: " + str(brick) + ". Next: " + str(mlnext_brick) + ". Prob: " + str(round(mlnext_prob,2)) + ".")


# In[12]:

def printNextBricks(brick_keys,probmat,numBricksToPrint,printbricks=True,printkeys=True,sortProb=False):
    """
    Print out the full list of brick transition probabilities for each brick.
    Also print out the list of key transition probabilities for each brick.
    """
    for i in range(50):
        #i = -i
        brick = brick_keys[i]
        print("Brick: "+str(brick)+".")
        row = probmat[i]
        nonzerolist = []
        for j in range(len(row)):
            if row[j] != 0:
                nonzerolist.append((row[j],brick_keys[j]))
        nonzerolist.sort(key=itemgetter(0),reverse=True)
        if printbricks:
            for nextBrick in nonzerolist:
                print("  Next: " + str(nextBrick[1]) + ". Prob: " + str(round(nextBrick[0],2)) + ".")
        bkeylist = [[i,0] for i in range(len(KEY_LABELS))]
        for nextBrick in nonzerolist:
            bkey = nextBrick[1]
            if len(bkey) != 2:
                printkeys = False
                break
            key = bkey[1]
            index = KEY_LABELS.index(key)
            bkeylist[index][1] += nextBrick[0]
        if printkeys:
            if sortProb:
                bkeylist.sort(key=itemgetter(1),reverse=True)
            for bkey in bkeylist:
                keyindex = bkey[0]
                keyprob = bkey[1]
                if keyprob != 0:
                    print("  Next key: " + KEY_LABELS[keyindex] + ". Prob: " + str(round(keyprob,2)))