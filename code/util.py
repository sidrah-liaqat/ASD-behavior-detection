from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import imageio
from PIL import Image, ImageDraw, ImageFont
w = 720
h = 480
ht = 30

def show_landmarks_pillow(sample, act_image):
    landmarks, eyelandmarks, headpose = sample['landmarks'], sample['eyelandmarks'], sample['headpose']
         #sample['landmarks'], sample['eyelandmarks'], sample['headpose'], sample['gazeangle']


    PIL_image = Image.fromarray(np.uint8(act_image))
    #PIL_image = Image.fromarray(255 * np.zeros((h, w, 3), np.uint8))
    """assuming img already RGB"""
    draw = ImageDraw.Draw(PIL_image, 'RGBA')

    draw.rectangle(((headpose[0, 0] * w, headpose[0, 1] * h),
                    (headpose[1, 0] * w, headpose[1, 1] * h)), width=4)

    # for drawing head ellipse in top right corner of head box
    """
    tlx = (headpose[1, 0] + headpose[0, 0]) / 2
    bry = (headpose[0, 1] + headpose[1, 1]) / 2
    draw.ellipse((tlx * w, headpose[0, 1] * h,
                  headpose[1, 0] * w, bry * h), fill=(255, 0, 0, 125))
    """
    # for drawing head ellipse in full head bounding box
    draw.ellipse((headpose[0, 0] * w, headpose[0, 1] * h,
                  headpose[1, 0] * w, headpose[1, 1] * h), fill=(255, 0, 0, 125))
    xy = list()
    #for i in range(eyelandmarks.shape[0]):
    #    xy.append((eyelandmarks[i, 0] * w, eyelandmarks[i, 1] * h))
    #draw.point(xy, fill=None)

    xy = list()
    for i in range(landmarks.shape[0]):
        xy.append((landmarks[i, 0] * w, landmarks[i, 1] * h))
    draw.point(xy, fill=None)

    tpi = torch.Tensor([np.pi])
    gaze_xy = list()
    # starting point is face landmark 27
    x1 = landmarks[27, 0] * w
    y1 = landmarks[27, 1] * h
    gaze_xy.append((x1, y1))
    # x2 = x1 + 20 * torch.sin(tpi*gazeangle[0]+tpi/2)
    # y2 = y1 + 20 * torch.sin(tpi*gazeangle[1]+tpi/2)
    # x2 = x1 + 100*np.sin(gazeangle[0])
    # y2 = y1 + 100*np.cos(gazeangle[1])
    #gazeangle = gazeangle.squeeze(0)
    #x2 = x1 + 100 * gazeangle[0]
    #y2 = y1 + 200 * gazeangle[1]

    draw.text((50, 50), 'Look Face', (255, 255, 255))
    #draw.text((500, 60), str(gazeangle[1]), (255, 255, 255))

    draw.line(gaze_xy, width=5)

    return PIL_image

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:,0], landmarks[:,1], s=4, marker='.', c='b')
    #TODO: Also draw rectangle for head
    plt.pause(0.3)  # pause a bit so that plots are updated

# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    #images_batch, lmk_x_batch, lmk_y_batch = \
    #        sample_batched['image'], sample_batched['lmk_x'], sample_batched['lmk_y']

    images_batch, landmarks_batch, headpose_batch = \
        sample_batched['image'], sample_batched['landmarks'], sample_batched['headpose']
    batch_size = len(images_batch)
    im_size = images_batch.size(3)
    w = images_batch.size(3)
    h = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)).astype(int))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy()*w + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy()*h + grid_border_size,
                    s=5, marker='.', c='b')

        plt.scatter(headpose_batch[i, : , 0].numpy() * w + i * im_size + (i + 1) * grid_border_size,
                    headpose_batch[i, : , 0].numpy() * h + grid_border_size,
                    s=10, marker='.', c='r')
    
        plt.title('Batch from dataloader')

def label_to_img(label, ht=30, color= [64, 128, 192, 125] ):

    # this function gives out one row of color
    # converts 1d data to 3D image like tensor
    w = len(label)
    a = np.array(label).astype(np.uint8)
    b = np.array(np.ones((ht), np.uint8))
    c = a[:, np.newaxis]*b
    c = np.transpose(c)

    img = c[:,:, np.newaxis]*color
    return img.astype(np.uint8)

def pred_to_stats(array):
    # This function takes frame by frame ground truth or prediction
    # (clustered, ideally) and converts it to duration and frequency
    # The objective is to compare the DL model predictions with
    # groundtruth statistics (duration and frequency of events)

    # array: 1D input numpy array consisting of ones and zeros

    dur = np.sum(array)/30.0
    freq = np.size(np.where(np.diff(array, prepend=0, append=0) == 1))
    return dur, freq

def make_fullimage(label1, label15, label2, label3):
    # this function takes a full length label and prediction
    # then forms image out of it
    # it is expected that label and prediction will be of same length
    fullimage = 255 * np.zeros((h+120+120, w, 3), np.uint8)
    # label color blue
    # prediction color red
    blue = (0, 0, 200)
    red = (255, 0, 0)
    green = (0, 200, 0)
    orange = (255, 215, 0)

    ind, rem = divmod(len(label1), w)
    i=0
    for i in range(ind):

        fullimage[i * 2 * ht + 0:i * 2 * ht + 15, :, :] = \
            label_to_img(label1[i * w:(i + 1) * w], ht=15,color=blue)
        fullimage[i * 2 * ht + 15:i * 2 * ht + 30, :, :] = \
            label_to_img(label15[i * w:(i + 1) * w], ht=15, color=red)
        fullimage[i * 2 * ht + 30:i * 2 * ht + 45, :, :] = \
            label_to_img(label2[i * w:(i + 1) * w], ht=15, color=green)
        fullimage[i * 2 * ht + 45:i * 2 * ht + 60, :, :] = \
            label_to_img(label3[i * w:(i + 1) * w], ht=15, color=orange)
    # drawing the remainder part of timeline
    if(ind>0):
        i += 1

    fullimage[i * 2 * ht + 0:i * 2 * ht + 15, 0:rem, :] = \
        label_to_img(label1[i * w:i*w+rem], ht=15, color=blue)
    fullimage[i * 2 * ht + 15:i * 2 * ht + 30, 0:rem, :] = \
        label_to_img(label15[i * w:i*w+rem], ht=15, color=red)
    fullimage[i * 2 * ht + 30:i * 2 * ht + 45, 0:rem, :] = \
        label_to_img(label2[i * w:i*w+rem], ht=15, color=green)
    fullimage[i * 2 * ht + 45:i * 2 * ht + 60, 0:rem, :] = \
        label_to_img(label3[i * w:i*w+rem], ht=15, color=orange)
    return fullimage

def mark_curr_frame(image, fidx):
    f_ind, f_rem = divmod(fidx, w)
    PIL_image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(PIL_image)
    draw.rectangle(((int(f_rem - 1), int(f_ind * 60)),
                    (int(f_rem + 1), int((f_ind + 1) * 60))), width=4)
    #draw.rectangle(((10, 10),
    #                (20, 20)), width=10)
    return PIL_image

def find_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.
    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.
    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """
    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2
    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections

#def slide_add(dest_arr, incoming, frame, i):
    # function takes in predicted_arr,
    #for idx in incoming.shape[1]:
        #if idx < frame:
    #dest_arr = torch.add(dest_arr[i:i+frame], incoming])


