import cv2
import numpy as np
import gym
from gym.spaces import Box

positions = [20, 40, 60]
current_position = 0
change_counter = 1


def _process_frame42(frame, variation):
    # print(frame.shape)
    frame = frame[34:34 + 160, :160, :]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))

    # if args.collect_images and img_counter <= args.num_collected_imgs:
    #     np.save('unit/datasets/breakout-' + args.variation + '/trainB/img_' + str(img_counter), frame)

    if variation == 'constant-rectangle':
        for i in range(2):
            for j in range(4):
                frame[i + 60][j + 60][0] = 0
                frame[i + 60][j + 60][1] = 191
                frame[i + 60][j + 60][2] = 255

    elif variation == 'moving-square':
        global change_counter
        global current_position
        for i in range(3):
            for j in range(3):
                frame[60 + i][positions[current_position] + j][0] = 255
                frame[60 + i][positions[current_position] + j][1] = 0
                frame[60 + i][positions[current_position] + j][2] = 0
        if (change_counter % 1000) == 0:
            change_counter = 1
            current_position += 1
            if current_position > 2:
                current_position = 0
        change_counter += 1

    elif variation == 'green-lines':
        for i in range(30, 77):
            if i % 8 == 0:
                j = i
                l = 4
                while j < 77 and l < 76:
                    frame[i][l][0] = 102
                    frame[i][l][1] = 203
                    frame[i][l][2] = 50
                    j += 1
                    l += 1

    elif variation == 'diagonals':
        for i in range(30, 77):
            if i % 8 == 0:
                j = i
                l = 4
                while j < 77 and l < 76:
                    frame[j][l][0] = 210
                    frame[j][l][1] = 203
                    frame[j][l][2] = 50
                    j += 1
                    l += 1
    elif variation == 'standard':
        pass
    else:
        raise NotImplementedError
    # frame = frame.astype(np.float32)
    # frame *= (1.0 / 255.0)
    # frame = np.moveaxis(frame, -1, 0)
    return frame


class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None, variation=None):
        super(AtariRescale42x42, self).__init__(env)
        self.variation = variation
        self.observation_space = Box(0.0, 1.0, [3, 80, 80], dtype=np.float32)

    def observation(self, observation):
        return _process_frame42(observation, self.variation)
