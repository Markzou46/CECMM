import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
import torchvision.transforms as transforms


class Augment:
    def __init__(self, eps=1e-5):
        self.eps = eps
        # Define a composition of video transformations for training
        self.video_transform = transforms.Compose([
            # Add relevant transformations here if needed
        ])
        # Define a composition of video transformations for testing
        self.test_video_transform = transforms.Compose([
            # Add relevant transformations here if needed
        ])

    # Apply transformations for training videos
    def train_video_transform(self, input):
        transformed_frames = self.video_transform(input)
        return transformed_frames

    # Apply transformations for testing videos
    def test_video_transform(self, input):
        transformed_frames = self.test_video_transform(input)
        return transformed_frames

    # Normalize input to the range [0,1]
    def norm01(self, input):
        return [(each - each.min()) / (each.max() - each.min()) if each.sum() > 5 else each for each in input]

    # Z-score normalization
    def z_score_normalization(self, input):
        return [(each - each.mean()) / each.std() if each.std() > 0 else each for each in input]

    # Normalize input and scale to 255
    def norm255(self, input):
        input = self.norm01(input)
        return [each * 255 for each in input]

    # Flip the input vertically (up-down) or horizontally (left-right)
    def flip(self, input, ud=False, lr=False):
        if ud and np.random.randint(2):
            input = [np.ascontiguousarray(np.flip(each, 1)) for each in input]
        if lr and np.random.randint(2):
            input = [np.ascontiguousarray(np.flip(each, 2)) for each in input]
        return input

    # Rotate the input randomly and resize
    def rotate(self, input):
        resize = lambda arr, size: F.interpolate(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0),
                                                 size=size, mode='trilinear', align_corners=False).squeeze(0).squeeze(
            0).numpy()
        shape = [each.shape for each in input]
        ang = np.random.rand() * 90
        input = [ndimage.rotate(each, angle=ang, axes=(1, 2), mode='constant', cval=0) for each in input]
        input = [resize(each, size) for each, size in zip(input, shape)]
        return input

    # Apply random gamma correction
    def sigma(self, input, scale=[0.6, 1.4]):
        sigma_factor = float((np.random.rand(1) + scale[0]) / (scale[1] - scale[0]))
        if np.random.randint(2):
            return input
        input = self.norm01(input)
        input = [np.power(each, sigma_factor) for each in input]
        return input

    # Generate pseudo optical flow
    def psudo_flow(self, input, dist_list=[1, 4, 7], with_ori=True):
        def _flow(arr, dist_list):
            result_list = [arr[np.newaxis, ...], ] if with_ori else []
            for each_dist in dist_list:
                dist_arr = np.pad(arr[each_dist:], ((0, each_dist), (0, 0), (0, 0)), mode='edge') - arr
                result_list.append(dist_arr[np.newaxis, ...])
            return np.concatenate(result_list, axis=0)

        return [_flow(each, dist_list) for each in input]


# Test the class with some random inputs
if __name__ == '__main__':
    print(Augment)

    # Generate random inputs
    input1 = torch.randn(160, 64, 64)
    input2 = torch.randn(160, 64, 64)

    # Initialize Augment class
    augment = Augment()

    # Apply video transformation for training
    output = augment.train_video_transform([input1, input2])
    print("Transformed video frames.")