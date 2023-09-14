from xml.dom import minidom
import os

HOME_DIR = os.getcwd()
# based on  https://github.com/bjornstenger/xml2yolo/blob/master/convert.py



def convert_xml2yolo(fname):
    lut={}
    lut["deer"] = "deer"
    lut["wild_boar"] = "wild_boar"

    xmldoc = minidom.parse(fname)
    itemlist = xmldoc.getElementsByTagName('object')
    size = xmldoc.getElementsByTagName('size')[0]
    width = int((size.getElementsByTagName('width')[0]).firstChild.data)
    height = int((size.getElementsByTagName('height')[0]).firstChild.data)
    item_list = []
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
            
        item_list.append((int(xmin), int(ymin), int(xmax), int(ymax)))
        
    return item_list