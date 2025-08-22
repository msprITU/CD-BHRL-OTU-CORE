import os
import random
import json
import sys

person_videos = ["ballet", "bicycle", "group1", "group2", "group3", "kitesurfing", "longboard", "person2", "person4", "person5",
"person7", "person14", "person17", "person19", "person20", "rollerman", "skiing", "sup", "tightrope", "warmup", "wingsuit"]

#car_videos = ["car1", "car2", "car3", "car6", "car8", "car9", "car16", "carchase", "f1", "following", "liverRun", "nissan", "sup", "volkswagen"]
#biker_videos = ["bicycle"] bike1? yamaha? horseride?
#dog_videos = ["dog", "freesbiedog"]
#cat_videos = ["cat1", "cat2"]
l = list(range(len(person_videos)))
random.shuffle(l)
position_seq = l[1 % len(l)]
print(person_videos[position_seq])

vot_test_ref = "/root/BHRL/vot_annotation/{}/vot_{}_test.json". format(person_videos[position_seq], person_videos[position_seq])
json_file_ref = open(vot_test_ref, "r")
json_data_ref = json.load(json_file_ref)

l = list(range(len(json_data_ref["images"])))
random.shuffle(l)
position_ref = l[1 % len(l)]
print(position_ref)

for target_seq in person_videos:
    vot_test_tar = "/root/BHRL/vot_annotation/{}/vot_{}_test.json". format(target_seq, target_seq)
    json_file_tar = open(vot_test_tar, "r")
    json_data_tar = json.load(json_file_tar)
    json_data_tar["images"].insert(0, json_data_ref["images"][position_ref])
    json_data_tar["annotations"].insert(0, json_data_ref["annotations"][position_ref])
    json_data_tar["annotations"][0]["image_id"] = 100000
    json_data_tar["annotations"][0]["id"] = 100000
    json_data_tar["images"][0]["id"] = 100000


    json_object = json.dumps(json_data_tar, indent=4)
    with open(vot_test_tar.replace(".json", "_random_ref_5.json"), "w") as outfile:
        outfile.write(json_object)

# find_random_seq_ref()