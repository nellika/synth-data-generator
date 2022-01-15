import os
import random
import json
import math
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation as R


class SynthDataGenerator:
    def __init__(self, inputPath="", outPath="") -> None:
        self.objs = dict()                                          # container for objects with their poses
        self.inputPath = inputPath                                  # input path relative to class root directory
        self.outPath = outPath                                      # output path relative to class root directory
        self.pref = ""                                              # prefix used to name output files {pref}{n}.png

        self.pixelWidth = 800
        self.pixelHeight = 800
        self.nearPlane = 0.01                                       # rather self.obj[nearPlane] param later
        self.farPlane = 40                                          # rather self.obj[farPlane] param later
        self.fov = 0.691111207008362                                # used only for random gen
        self.rotation = 0.012566370614359171                        # not used

        self.physicsClient = pb.connect(pb.DIRECT)

    # def is_rotation_matrix(self, R):
    #     Rt = np.transpose(R)
    #     shouldBeIdentity = np.dot(Rt, R)
    #     I = np.identity(3, dtype = R.dtype)
    #     n = np.linalg.norm(I - shouldBeIdentity)
    #     return n < 1e-6

    # def rotation_matrix_to_euler(self, R):
    #     assert(self.is_rotation_matrix(R))

    #     sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    #     singular = sy < 1e-6

    #     if  not singular :
    #         x = math.atan2(R[2,1] , R[2,2])
    #         y = math.atan2(-R[2,0], sy)
    #         z = math.atan2(R[1,0], R[0,0])
    #     else :
    #         x = math.atan2(-R[1,2], R[1,1])
    #         y = math.atan2(-R[2,0], sy)
    #         z = 0

    #     return np.array([x, y, z])

    def read_poses(self, obj_name, file_name) -> None:
        """Reads transformation json into class variable from given input path"""
        with open(os.path.join(self.inputPath, obj_name, file_name), 'r') as f:
            full_json = json.load(f)
            
            self.objs[obj_name]['cameraAngleX'] = math.degrees(full_json['camera_angle_x'])

            frames = full_json['frames']
            for f in frames:
                trans_m = np.array(f['transform_matrix']).astype(np.float32)
                self.objs[obj_name]['camTrans'].append(np.transpose(np.linalg.inv(trans_m)))

    def create_transforms_file(self, file_name, obj_name, random_poses) -> None:
        """Random poses: creates transformation json"""
        transforms = dict()
        transforms['camera_angle_x'] = self.objs[obj_name]['cameraAngleX']
        transforms['frames'] = []
        i = 0

        for rnd_pose in random_poses:
            frame = dict()
            frame['file_path'] = self.pref + str(i)
            frame['rotation'] = self.rotation
            frame['transform_matrix'] = rnd_pose.tolist()
            transforms['frames'].append(frame)
            i+=1

        with open(os.path.join(self.outPath, obj_name, file_name + '.json'), 'w') as outfile:
            json.dump(transforms, outfile)


    def init_objects_and_poses(self, input_obj_names=None, pose_file_name=None, random_poses=False, nr_random_poses=7, init_orientation=None):
        """Object initialization, reading poses"""
        if ( not random_poses ) and ( pose_file_name == None ): return -1
        if ( init_orientation != None ) and len(input_obj_names) != len(init_orientation): return -1

        if input_obj_names != None:
            for o in range(0,len(input_obj_names)):
                obj_name = input_obj_names[o]

                self.objs[obj_name] = dict()
                self.objs[obj_name]["camTrans"] = []

                if init_orientation != None: self.objs[obj_name]["initOrn"] = init_orientation[o]
                else: self.objs[obj_name]["initOrn"] = [0,0,0]
                if random_poses:
                    self.objs[obj_name]['cameraAngleX'] = math.degrees(self.fov)
                    poses = self.get_random_poses(nr_random_poses)
                    for rnd_pose in poses:
                        self.objs[obj_name]["camTrans"].append(np.transpose(np.linalg.inv(rnd_pose)))
                    self.create_transforms_file("random_transforms", obj_name, poses)
                else:
                    self.read_poses(obj_name, pose_file_name)

    # def randomMatrix(self):
    #     rot = R.random()
    #     return rot

    # def translationMatrix(self, trans_min, trans_max, dim=3):
    #     return np.random.uniform(low=trans_min, high=trans_max, size=(dim,1))

    # def fullMatrix(self, rot, translation):
    #     m = np.hstack((rot, translation))
    #     last_row = [0,0,0,1]
    #     m = np.vstack((m, last_row))
    #     return m

    # def randomRotationMatrix(self, trans_min, trans_max):
    #     random_matrix = []
    #     rot = self.randomMatrix()
    #     yaw_pitch_roll = rot.as_euler('zyx')
    #     translation = self.translationMatrix(trans_min, trans_max)
    #     full_matrix = self.fullMatrix(rot.as_matrix(), translation)
    #     random_matrix.append(np.array(full_matrix))
    #     random_matrix.append(np.array(yaw_pitch_roll))
    #     random_matrix.append(np.array(translation))
    #     # print(random_matrix)
    #     random_matrix = np.array(random_matrix)
    #     return random_matrix

    # def get_random_poses(self, nr_samples, trans_min, trans_max):
    #     random_matrices = []
    #     for _ in range(nr_samples):
    #         random_matrices.append(self.randomRotationMatrix(trans_min, trans_max))
    #     return np.array(random_matrices)

    def get_random_trans(self):
        """Returns a set of random transformations based on the file rand_json in the root folder"""
        with open('rnd_trans.json', 'r') as f:
            full_json = json.load(f)
        return full_json["trans"]

    def get_random_poses(self, nr_samples,):
        """Returns a specific number of transformation matrices in a randomized fashion"""
        trans = self.get_random_trans()
        idx = random.sample(range(0, len(trans)), nr_samples)
        random_poses = [trans[t] for t in idx]
        return np.array(random_poses)

    def generate_images_from_cam_poses(self, obj_name, transparent = True):
        """Based on class parameters, generates poses, renders and saves images for a single object"""
        full_path = os.path.join(self.inputPath, obj_name, obj_name + '.urdf')

        init_orn = self.objs[obj_name]['initOrn']
        init_orn_quat = pb.getQuaternionFromEuler(init_orn)
        obj_id = pb.loadURDF(full_path, baseOrientation=init_orn_quat)

        fov = self.objs[obj_name]['cameraAngleX']

        # TODO: find appropriate material, maybe randomize, or find a way to apply mtl colours
        textureId = pb.loadTexture('input/thumb-Brick-0191.jpg')
        pb.changeVisualShape(obj_id, -1, textureUniqueId=textureId)

        i = 0
        for trans in self.objs[obj_name]['camTrans']:

            viewMatrix = trans.flatten()

            aspect = self.pixelWidth / self.pixelHeight
            projectionMatrix = pb.computeProjectionMatrixFOV(fov=fov,
                                                             aspect=aspect,
                                                             nearVal=self.nearPlane,
                                                             farVal=self.farPlane)

            _, _, rgbImg, _, segImg = pb.getCameraImage(self.pixelWidth,
                                                self.pixelHeight,
                                                viewMatrix,
                                                projectionMatrix,
                                                lightDirection=[1, 1, 1],
                                                renderer=pb.ER_BULLET_HARDWARE_OPENGL)

            if transparent:
                segImg = segImg + 1
                alpha = np.multiply(rgbImg[:,:,3],segImg)
                transpImg = rgbImg.copy()
                transpImg[:,:,3] = alpha
                plt.imsave(f'out/{obj_name}/{self.pref}{i}.png',transpImg)
            else: plt.imsave(f'out/{obj_name}/{self.pref}{i}.png',rgbImg)

            print(f'{obj_name} >> {self.pref}{i}.png done.')

            i+=1

        pb.removeBody(obj_id)

    def generate_images_from_objects(self, pref=""):
        self.pref = pref
        """Generates images based on previously initialized parameters for each object"""
        for obj_name in self.objs:

            directory_exists = os.path.isdir(os.path.join(self.outPath, obj_name))
            if not directory_exists: os.mkdir(os.path.join(self.outPath,obj_name))

            self.generate_images_from_cam_poses(obj_name)