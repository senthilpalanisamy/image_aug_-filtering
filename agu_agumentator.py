import Augmentor
import cv2
p = Augmentor.Pipeline("/home/guru/Desktop/Project_Prework/cap")
# Point to a directory containing ground truth data.
# Images with the same file names will be added as ground truth data
# and augmented in parallel to the original data.
#p.ground_truth("/home/guru/Downloads/cap.JPG")
# Add operations to the pipeline as normal:
p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.flip_top_bottom(probability=0.5)
p.rotate90(probability=0.5)
p.rotate270(probability=0.5)
p.flip_left_right(probability=0.8)
p.flip_top_bottom(probability=0.3)
p.crop_random(probability=1, percentage_area=0.5)
p.resize(probability=1.0, width=120, height=120)
# p.sample(50)
g = p.keras_generator(batch_size=128)
images, labels = next(g)
#cv2.imwrite('../cap/cap%s.jpg'%i,img)