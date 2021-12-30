import functools
import argparse
import os
from tqdm import tqdm
from matplotlib import gridspec
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub

def _crop_center(image):
	"""Returns a cropped square image."""
	shape = image.shape
	new_shape = min(shape[1], shape[2])
	offset_y = max(shape[1] - shape[2], 0) // 2
	offset_x = max(shape[2] - shape[1], 0) // 2
	return tf.image.crop_to_bounding_box(image, offset_y, offset_x, new_shape, new_shape)

@functools.lru_cache(maxsize=None)
def _load_image(image_url=None, image_size=(256, 256), preserve_aspect_ratio=True):
	"""Loads and preprocesses images."""
	if os.path.exists(image_url):
		image_path = image_url
	else:
		# Cache image file locally.
		image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)

	# Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
	img = tf.io.decode_image(
		tf.io.read_file(image_path),
		channels=3, dtype=tf.float32)[tf.newaxis, ...]
	img = _crop_center(img)
	return tf.image.resize(img, image_size, preserve_aspect_ratio=preserve_aspect_ratio)

def get_content_tensor(image_url):
    image = _load_image(image_url=image_url, image_size=(384, 384))
    return tf.constant(image)

def get_style_tensor(image_url):
    image = _load_image(image_url=image_url, image_size=(256, 256)) # Recommended to keep it at 256.
    image = tf.nn.avg_pool(image, ksize=[3,3], strides=[1,1], padding='SAME')
    return tf.constant(image)

def show_n(images, titles=['Content', 'Style', 'Stylized image']):
	n = len(images)
	image_sizes = [image.shape[1] for image in images]
	w = (image_sizes[0] * 6) // 320
	plt.figure(figsize=(w * n, w))
	gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
	for i in range(n):
		plt.subplot(gs[i])
		plt.imshow(images[i][0], aspect='equal')
		plt.axis('off')
		plt.title(titles[i] if len(titles) > i else '')
	plt.show()

def main():
    # argparse initialization
    parser = argparse.ArgumentParser(description='Styleze specified images.')
    parser.add_argument('path_to_content', help='Content image')
    parser.add_argument('path_to_style', help='Style image')
    parser.add_argument('--save', help='Save or not', action='store_true')
    args = parser.parse_args()

	# load hub_module
    # hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    hub_module = tf.saved_model.load('./hub_module')

	# Stylization
    with tqdm(desc='Stylizing', total=3, ncols=100) as pbar:
        pbar.update(1)

        content_name = args.path_to_content # 'hu-goabakio.jpg'
        style_name   = args.path_to_style # 'josuke4.png'
        content_img  = get_content_tensor(content_name)
        style_img    = get_style_tensor(style_name)
        pbar.update(1)
        
        stylized_img = hub_module(content_img, style_img)[0]
        pbar.update(1)

	# show results
    show_n([content_img, style_img, stylized_img])
    
    # save the result image
    if args.save:
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(stylized_img[0])
        plt.axis('off')
        plt.close(fig)
        fig.savefig('out.png', bbox_inches='tight', pad_inches=0)
        print('Output saved correctly.')
    
if __name__ == '__main__':
    main()