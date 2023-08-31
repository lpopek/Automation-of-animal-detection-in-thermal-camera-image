from xml.dom import minidom
import os
import glob

HOME_DIR = os.getcwd()
# based on  https://github.com/bjornstenger/xml2yolo/blob/master/convert.py
lut={}
lut["deer"] = "deer"
lut["wild_boar"] = "wild_boar"


def convert_xml2yolo(lut, fname, output_dir):

    xmldoc = minidom.parse(fname)
    
    fname_out = (output_dir + fname[:-4]+'.txt')

    with open(fname_out, "w") as f:

        itemlist = xmldoc.getElementsByTagName('object')
        size = xmldoc.getElementsByTagName('size')[0]
        width = int((size.getElementsByTagName('width')[0]).firstChild.data)
        height = int((size.getElementsByTagName('height')[0]).firstChild.data)

        for item in itemlist:
            # get class label
            classid =  (item.getElementsByTagName('name')[0]).firstChild.data
            if classid in lut:
                label_str = str(lut[classid])
            else:
                label_str = "-1"
                print ("warning: label '%s' not in look-up table" % classid)

            # get bbox coordinates
            xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
            ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
            xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
            ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
            b = (int(xmin), int(ymin), int(xmax), int(ymax))
            #print(bb)

            f.write(label_str + " " + " ".join([str(a) for a in b]) + '\n')

    print ("wrote %s" % fname_out)



def main():
    os.chdir("./test-mAP/predicted_boxes")
    file_list = os.listdir()
    os.chdir(HOME_DIR + "/dataset_update/")
    for item in file_list:
        if "deer" in item: 
            os.chdir("deer/xml/")
            fname = item.replace("txt", "xml")
            convert_xml2yolo(lut, fname, HOME_DIR + "/test-mAP/ground_truth/")
        else:
            os.chdir("wild_boar/xml/")
            fname = item.replace("txt", "xml")
            convert_xml2yolo(lut, fname, HOME_DIR + "/test-mAP/ground_truth/")
        os.chdir(HOME_DIR + "/dataset_update/")


if __name__ == '__main__':
    main()