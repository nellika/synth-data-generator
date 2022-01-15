from numpy import random
import SynthDataGenerator as sdg
import math


input_path = "input"
out_path = "out"
data_generator = sdg.SynthDataGenerator(input_path, out_path)

# obj_names = ["Lego_856_Bulldozer", "duck_vhacd"]
obj_names = ["Lego_856_Bulldozer_2", "arm_tr", "duck_vhacd"]
init_orientation = [[math.pi/2,0,0.012566370614359],[0,0,0], [0,0,0]]
pose_file_name = "transforms.json"


# data_generator.init_objects_and_poses(input_obj_names=obj_names, pose_file_name=pose_file_name, init_orientation=init_orientation)
# data_generator.generate_images_from_objects()

# TODO: define multiple random sets, and pick from them uniformely (to have img from major angles)
data_generator.init_objects_and_poses(input_obj_names=obj_names, random_poses=True, nr_random_poses=7)
data_generator.generate_images_from_objects(pref="rnd_")
