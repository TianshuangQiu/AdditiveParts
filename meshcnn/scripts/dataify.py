import ijson
import os
import random
import shutil
import numpy as np
# with open("./data/merged_data.json", "rb") as f:
#         parser = ijson.parse(f)
#         scores = []
#         for file in parser:
#                 if (file[1] == 'string'):
#                         key, value = file[0], file[-1]
#                         name = os.path.basename(key)[:-4] + "_simplified.obj"
#                         location = os.path.join("./scripts/simplified", "output", name)

#                         if (not os.path.isfile(location)):
#                                 continue
#                         scores.append(min(100.0, float(value)))c
with open("./data/merged_data.json", "rb") as f:
        parser = ijson.parse(f)
        for file in parser:
                if (file[1] == 'string'):
                        key, value = file[0], file[-1]
                        name = os.path.basename(key)[:-4] + "_simplified.obj"
                        location = os.path.join("./scripts/simplified", "output", name)

                        if (not os.path.isfile(location)):
                                continue

                        # does train test split
                        
                        score = float(value)
                        # we ball, make the classes I guess      
                        new_dir = str(min(score, 2))
                        if (not os.path.exists(os.path.join("./dataset/sdata", new_dir))):
                                os.mkdir(os.path.join("./dataset/sdata", new_dir))
                                print("HELLO???")
                                if (not os.path.exists(os.path.join("./dataset/sdata", new_dir, "test"))):
                                        os.mkdir(os.path.join("./dataset/sdata", new_dir, "test"))
                                        os.mkdir(os.path.join("./dataset/sdata", new_dir, "train"))
                                if (not os.path.exists(os.path.join("./dataset/sdata", new_dir, "train"))):
                                        os.mkdir(os.path.join("./dataset/sdata", new_dir, "train"))
                                        os.mkdir(os.path.join("./dataset/sdata", new_dir, "train"))
                        if (random.random() < 0.2):
                                newhome = os.path.join("./dataset/sdata", new_dir, "test", name)
                                shutil.move(location, newhome)
                                # print("new file test", newhome)
                        else:
                                newhome = os.path.join("./dataset/sdata", new_dir, "train", name)
                                shutil.move(location, newhome)
                                # print("new file train", newhome)
