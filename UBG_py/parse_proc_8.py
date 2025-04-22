import json
import numpy as np
proc_json_name = "D:/python_workspace/blender_use/procdata/train_"+"5"+".json"
import os
def find_min_x_y(points):
    if not points:
        return None, None

    min_x = points[0][0]
    min_y = points[0][1]

    for x, y in points:
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y

    return min_x, min_y


def find_centroid(points):
    if not points:
        return None  # 如果点列表为空，返回None

    sum_x = sum(point[0] for point in points)
    sum_y = sum(point[1] for point in points)
    num_points = len(points)

    centroid_x = sum_x / num_points
    centroid_y = sum_y / num_points

    return (centroid_x, centroid_y)

def contains_digit(s):
    return any(char.isdigit() for char in s)
def parse_object_ID(input_data):
    split_data = input_data.split("_")
    parse_ID2folder = ""
    for i in split_data:
        if contains_digit(i):
            break
        else:
            parse_ID2folder += i
            parse_ID2folder += "_"
    return parse_ID2folder[:-1]

def parse_folder_name(father_path):
    all_folder_name = []
    for entry in os.scandir(father_path):
        if entry.is_dir():
            all_folder_name.append(entry.name)
    return all_folder_name

with open(proc_json_name, 'r', encoding='utf-8') as file:
    data = json.load(file)
    object_scale = np.array([0.01, 0.01, 0.01])

    all_room = data['rooms']
    all_room_verticle = []
    for room in all_room:
        polygon_vertical = room['floorPolygon']
        for i in polygon_vertical:
            all_room_verticle.append((i['z'], i['x']))
    room_min_x, room_min_y = find_min_x_y(all_room_verticle)
    all_object = data['objects']
    object_count = 0
    object_count_child = 0
    folder_path = "E:/usd_aaset/ExportedUSD/"
    all_folder = parse_folder_name(folder_path)
    for each_object in all_object:
        each_object_id = each_object['assetId']
        each_object_kinematic = each_object['kinematic']
        parsed_name = parse_object_ID(each_object_id)
        last_find_object_usd = ""
        if "RoboTHOR" in parsed_name:
            object_count += 1
            last_find_object_usd = folder_path + "RoboTHOR/" + each_object_id + ".usdz"
        else:
            object_count += 1
            for folder_search in all_folder:
                if parsed_name == folder_search:
                    last_find_object_usd = folder_path + parsed_name + "/" + each_object_id + ".usdz"
                    break
        object_pos = each_object['position']
        object_rot = each_object['rotation']
        if "Decor" in each_object_id:
            use_pos = np.array([object_pos['z'], -object_pos['x'], 2.4])
        else:
            if "Countertop_L_6x6" in each_object_id:
                if "children" in each_object:
                    all_pos_child = []
                    each_object_child = each_object['children']
                    for each_object_child_2 in each_object_child:
                        parse_pos_child_2 = each_object_child_2['position']
                        all_pos_child.append((parse_pos_child_2['z'], -parse_pos_child_2['x']))
                    centroid_child = find_centroid(all_pos_child)
                    centroid_counter = (object_pos['z'], -object_pos['x'])
                    print("center point child  ", centroid_child)
                    print("counter point center ", centroid_counter)
                    cha_x = centroid_counter[0] - centroid_child[0]
                    cha_y = centroid_counter[1] - centroid_child[1]
                    print("cha x  ", centroid_counter[0] - centroid_child[0])
                    print("cha y  ", centroid_counter[1] - centroid_child[1])
                    use_pos = np.array([object_pos['z'], object_pos['x'], 0])
                    if centroid_child[0] >= centroid_counter[0]:
                        use_pos[0] = use_pos[0] + 0.9145
                    else:
                        use_pos[0] = use_pos[0] - 0.9145
                    if centroid_child[1] >= centroid_counter[1]:
                        use_pos[1] = use_pos[1] + 0.94
                    else:
                        use_pos[1] = use_pos[1] - 0.94
                # print(object_count_child,"child~~~~~~~~~~~~~~~~~~~~~~~",each_object_child_2_id)
                # if object_pos['z'] >= 0:
                #     object_pos['z'] = object_pos['z']+0.9145
                # else:
                #     object_pos['z'] = object_pos['z']-0.9145
                # if object_pos['x'] >= 0:
                #     object_pos['x'] = object_pos['x']+0.939
                # else:
                #     object_pos['x'] = object_pos['x']-0.939
            else:
                use_pos = np.array([object_pos['z'], -object_pos['x'], 0])
        use_pos = [use_pos[0] , use_pos[1] , use_pos[2]]
        print(last_find_object_usd)
        unity_export_rot_x = object_rot['x'] + 90.0
        unity_export_rot_y = -object_rot['y'] - 90.0
        unity_export_rot_z = object_rot['z']
        use_rot = [unity_export_rot_x, unity_export_rot_z, unity_export_rot_y]
        # print(last_find_object_usd,"  ~~   ",use_pos,use_rot)