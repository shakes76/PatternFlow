# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 30/09/2019
import numpy as np
import imageio
import sys
from tabulate import tabulate

import vif

##Driver script
if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise NameError('Missing arguments: needs one reference image and at least one image to compare')
    else:
        name_ref = sys.argv[1]
        name_query = sys.argv[2:]

    img_ref = imageio.imread(name_ref, as_gray=True).astype(np.float32)
    img_query = []
    for i in range(0, len(name_query)):
        img_query.append(imageio.imread(name_query[i], as_gray=True).astype(np.float32))
    
    vif_tab = vif.pbvif(img_ref, img_query)

    print(tabulate([(name_query[i], vif_tab[i]) for i in range(len(name_query))], headers=['Image', 'pbvif'], tablefmt='github'))
