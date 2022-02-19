import numpy as np
import cv2
from imutils.video import FPS
import os

# -------------------------------------------------------------------------------------------
# --------------------------------DIFFERENT METHODS------------------------------------------
# -------------------------------------------------------------------------------------------

# Get the Main color in the bounding box and the precision
# image = an image ; colors_on_the_fild = list with all the colors to analyse (ex : ["White","Blue","Red"])
def PrimeColor(image, colors_on_the_field):
    #cv2.imshow('image',image)
    max_ratio = 0
    final_color = ""
    confidence = 0

    # FIXME : Add Cyan
    color_list = ['Red', 'Blue', 'White', 'Black', 'Yellow', 'Green', 'Purple']
    boundaries = [
        ([0, 100, 100], [10, 255, 255]),  # red
        ([110, 50, 50], [130, 255, 255]),  # blue
        ([0, 0, 213], [255, 20, 255]),  # white
        ([0, 0, 0], [10, 10, 10]),  # black
        ([80, 50, 50], [100, 255, 255]),  # yellow
        ([50, 50, 50], [70, 255, 255]),  # green
        ([140, 50, 50], [160, 255, 255]),  # purple (Magenta)
    ]

    try:
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        for color in colors_on_the_field:
            if color not in color_list:
                raise ValueError("This color : {}, is not yet defined in our code ... Sorry :( ".format(color))
            else:
                index = color_list.index(color)
                lower_range = np.array(boundaries[index][0])
                upper_range = np.array(boundaries[index][1])
                mask = cv2.inRange(hsv, lower_range, upper_range)

                ratio = cv2.countNonZero(mask) / np.size(mask)
                confidence += ratio
                if (ratio > max_ratio):
                    max_ratio = ratio
                    final_color = color

        try:
            conf = max_ratio / confidence
        except:
            
            final_color = "ERROR"
            conf = 0
    except:
        conf = 0
        final_color = "ERROR"
        

    return final_color, conf


## Get an array with all the detections present in the detection_file
# f = detection_file (string like 'mydetections.txt')
def convert(f):
    file = open(f, 'r')
    frame_detections = []
    for line in file.readlines():
        detection = line[:-1].split(',')
        frame_detections.append(detection)
    return frame_detections


# Pass through the video and the detection_file to detect the color of each detections
def find_color_of_ids(INPUT_FILE, detection_file, colors_on_the_field, precision):
    # INPUT_FILE is the frames directory of the analyzed video
    # detection_file is the .txt with all the detections
    # colors on the field are all the colors of the shirts of the teams, referee, goalkeeper ...
    # precision is the limit precision to validate a color.

    # This fonction will return a list with all the detections and added 2 columns : Player's color and precision of
    # color
    all_detections = convert(detection_file)
    cnt = 0
    index = 0

    # a new id is created
    IDs = []  # Variable that stores all the Ids we've seen yet
    num_col_id = []
    video = cv2.VideoCapture(INPUT_FILE)
    while(video.isOpened()):
        ret, image = video.read()
        cnt += 1
        print("Frame number", cnt)

        # We take the boxes ans IDs of the OL_PSG file
        detections = []
        try:
            while (int(all_detections[index][0]) == cnt):
                detections.append(all_detections[index])
                index += 1
        except:
            print("ERROR")
            break

        for detection in detections:
            # extract the bounding box coordinates
            (x, y) = (round(float(detection[2])), round(float(detection[3])))
            (w, h) = (round(float(detection[4])), round(float(detection[5])))

            # Now i want to detect which color is majoritaire in each bounding boxes:
            crop_img = image[y:y + h, x:x + w]
            main_color, conf_color = PrimeColor(crop_img, colors_on_the_field)
            print(main_color," ",conf_color)

            if (int(detection[1]) not in IDs):
                IDs.append(int(detection[1]))
                num_col_id.append({color: 0 for color in colors_on_the_field})

            if (main_color != "ERROR"):
                if (conf_color > precision):
                    try:
                        num_col_id[int(detection[1]) - 1][main_color] += 1
                    except:
                        print("Some Ids were skipped")
                        for i in range(int(detection[1]) - len(num_col_id)):
                            num_col_id.append({color: 0 for color in colors_on_the_field})
                        num_col_id[int(detection[1]) - 1][main_color] += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # do a bit of cleanup
    video.release()
    cv2.destroyAllWindows()
    

    # release the file pointers
    print("[INFO] cleaning up...")

    count = 0
    '''for elmnt in num_col_id:
        print(count, "  ",elmnt)
        count+=1'''
    create_and_save_tab_with_colors(num_col_id, all_detections, colors_on_the_field, output_file=detection_file+"_colored")
    return num_col_id


# Pass through the video and the detection_file to detect the color of each detections naïve version (with historic of 10 colors)
def find_color_of_ids_naive(INPUT_FILE, detection_file, colors_on_the_field, precision, historic_size=10):
    # INPUT_FILE is the video we are analysing
    # detection_file is the .txt with all the detections
    # colors on the field are all the colors of the shirts of the teams, arbitre, goal ...
    # precision is the limit precision to validate a color.

    # This fonction will return a list with all the detections and added 2 columns : Player's color and precision of color
    
    all_detections = convert(detection_file)

    cnt = 0
    index = 0

    # a new id is created
    IDs = []  # Variable that stocks all the Ids we've seen yet

    historic = []
    # We empty the file in case it already exists
    open(detection_file+'_colored', 'w').close()
    video = cv2.VideoCapture(INPUT_FILE)
    while(video.isOpened()):
        ret, image = video.read()
        cnt += 1
        print("Frame number", cnt)

        # We take the boxes ans IDs of the OL_PSG file
        detections = []
        try:
            while (int(all_detections[index][0]) == cnt):
                detections.append(all_detections[index])
                #FIXME : To much appending in the historic list
                historic.append([])
                index += 1
        except:
            break

        for detection in detections:
            if(int(detection[-1]) == 1):
                # extract the bounding box coordinates
                (x, y) = (round(float(detection[2])), round(float(detection[3])))
                (w, h) = (round(float(detection[4])), round(float(detection[5])))

                # Now i want to detect which color is majoritaire in each bounding boxes:
                crop_img = image[y:y + h, x:x + w]
                main_color, conf_color = PrimeColor(crop_img, colors_on_the_field)
                #print(main_color," ",conf_color)

                if (main_color != "ERROR"):
                    # We add the ID's as we meet them
                    if (int(detection[1]) not in IDs):
                        IDs.append(int(detection[1]))
                        historic.append([])

                    if (conf_color > precision):
                        try:
                            # We add the color detected to the historic of the ids
                            historic[int(detection[1])].append(main_color)
                        except:
                        
                            for i in range(int(detection[1]) - len(historic)):
                                historic.append([])
                            historic[int(detection[1])].append(main_color)

                # And then we choose the color that is the most present in the historic
                if (len(historic[int(detection[1])]) != 0):
                    addobservation(detection, most_frequent(historic[int(detection[1])]), output=detection_file+'_colored')

                if (len(historic[int(detection[1])]) > historic_size):
                    # We limit the historic to ten
                    historic[int(detection[1])].pop(0)
            else : 
                addobservation(detection, 'Black', output=detection_file+'_colored')



        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    i = 0

def addobservation(detection,color,output):
    with open(output, 'a') as f:
        for item in detection:
            f.write(item+",")
        f.write(color+"\n")

#Finds the most frequent element in a list
def most_frequent(mylist):
    mymax=0
    res = ""
    for elmt in mylist:
        count = mylist.count(elmt)
        if (count > mymax):
            mymax = count
            res = elmt
    return res

## Show the bounding box with the id detected by footAndBall
def test_our_teams(INPUT_FILE, detection_file):
    # INPUT_FILE='MyPart/extraitOM-OL.mp4'
    OUTPUT_FILE = 'output.mov'

    fps = FPS().start()

    all_detections = convert(detection_file)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, 30,
                             (800, 600), True)

    vs = cv2.VideoCapture(INPUT_FILE)
    cnt = 0
    index = 0

    # We want to know the number of time the player was identified white or blue
    # Would be change beacause we don't know how many ids will there be.
    # We would just have to append a new {"White" : 0,"Blue" : 0 ,"Red":0} each time
    # a new id is created
    num_col_id = [{"White": 0, "Blue": 0, "Red": 0} for _ in range(164)]

    (grabbed, image) = vs.read()
    while True:
        cnt += 1
        print("Frame number", cnt)
        try:
            (grabbed, image) = vs.read()
        except:
            break

        # We take the boxes ans IDs of the OL_PSG file
        detections = []
        try:
            while (int(all_detections[index][0]) == cnt):
                detections.append(all_detections[index])
                index += 1
        except:
            break

        for detection in detections:
            # extract the bounding box coordinates
            (x, y) = (round(float(detection[2])), round(float(detection[3])))
            (w, h) = (round(float(detection[4])), round(float(detection[5])))

            # Now i want to detect which color is majoritaire in each bounding boxes:
            # I need a function that detect the Prime Color in a frame:
            # crop_img = image[y+int(h/5):y+int(2*h/3), x+int(w/3):x+w]
            crop_img = image[y:y + h, x:x + w]

            main_color = detection[-1]

            if main_color == "Red":
                ## METTRE EN ARG HLOBAL LE SEUIL
                color = (0, 0, 255)

            elif main_color == "Blue":
                color = (255, 0, 0)
                # main_color = "white"


            elif main_color == "White":
                color = (255, 255, 255)


            else:
                color = (0, 0, 0)

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "ID = {} /".format(detection[1])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        # show the output image
        cv2.imshow("output", cv2.resize(image, (800, 600)))
        writer.write(cv2.resize(image, (800, 600)))
        fps.update()

        # time.sleep(1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()

    # release the file pointers
    print("[INFO] cleaning up...")
    writer.release()
    vs.release()


## Save and write a file with all the detections + the colors
def create_and_save_tab_with_colors(ids_colors_attribution, detection_tab, colors_on_the_field, output_file):
    new_tab = []
    id_final_colors = []

    # Here we loop to get the color that is the most assigned to each IDs
    for element in ids_colors_attribution:
        max_col = ""
        max_value = 0
        for col in colors_on_the_field:
            if (element[col] > max_value):
                max_col = col
                max_value = element[col]

        if (max_value == 0):  ## If no colors was ever detected for this id write ERROR
            max_col = "ERROR"

        id_final_colors.append(max_col)

    for detection in detection_tab:
        new_tab.append(detection + [id_final_colors[int(detection[1]) - 1]])

    with open(output_file, 'w') as f:
        for line in new_tab:
            for item in line[:-1]:
                f.write(item + ",")
            f.write(line[-1])
            f.write("\n")
        return new_tab


# -------------------------------------------------------------------------------------------
# --------------------------------DIFFERENT METHODS------------------------------------------
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
# --------------------------------MAIN-PROGRAMME---------------------------------------------
# -------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("coucou")
    # CREATE A FILE WITH THE COLORS OF EACH PLAYERS
    # Try changing the INPUT_FILE (.mov or .mp4) and the detection_file that corresponds,
    # don't forget to change the colors present on the field
    # --------------------------------------------------------------------------------------
    # find_color_of_ids('extraitOM-OL.mp4', "OL_PSG", ['Blue', 'White', 'Red'], 0.8)
    # --------------------------------------------------------------------------------------

    # See the result on a OUTPUT_FILE to see the players live with the bounding box colored in the right color
    # --------------------------------------------------------------------------------------
    # test_our_teams('extraitOM-OL.mp4', "myColoredDetection.txt")
    # --------------------------------------------------------------------------------------
    

# -------------------------------------------------------------------------------------------
# --------------------------------MAIN-PROGRAMME---------------------------------------------
# -------------------------------------------------------------------------------------------
