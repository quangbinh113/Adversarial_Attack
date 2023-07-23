import os
import sys
import csv
from skimage import io
from skimage.metrics import structural_similarity as ssim
from AttackFace import tools, imagelabels, attacks
from PIL import Image


def draw_img_noise_pair_arrays(image_noise_pairs, epsilon, img_num, output_dir, use_mult_noise=False):

    for test_num, img_pair in enumerate(image_noise_pairs):
        attacked_img = img_pair.img
        noise_img = img_pair.noise
        file_suffix = str(test_num) +'.jpg'

        image_file_name = file_suffix
        image_file_path = os.path.join(output_dir,image_file_name)
        im_pix = Image.fromarray(attacked_img)
        im_pix.save(image_file_path)



def get_ssim(image_labels, attacked_image_labels):
    """
    Given an unattacked and an attacked ImageLabels objects, calculate the SSIM score between them
    
    Args:
        image_labels: the ImageLabels object for the unattacked image
        attacked_imaged_labels: the ImageLabels object for the attacked image
    Returns:
        ssim_val: the SSIM value calculated between the two images
    """
    img = io.imread(image_labels.img_path)
    attack_img = io.imread(attacked_image_labels.img_path)
    ssim_val = ssim(
            img,
            attack_img,
            data_range = attack_img.max() - attack_img.min(),
            multichannel=True)
    return ssim_val


def write_csv(rows, output_filename, output_dir):

    if (len(rows) != 0):
        file_path = os.path.join(output_dir, output_filename)
        with open(file_path, 'w', encoding='utf8', newline='') as output_file:
            if (isinstance(rows, dict)): rows = [rows]
            fc = csv.DictWriter(output_file,fieldnames=rows[0].keys())
            fc.writeheader()
            fc.writerows(rows)
    else: 
        print('write_csv \'rows\' argument has no entries; cannot write CSV file: ' 
            + str(os.path.join(output_dir, attack_record_filename)))


def get_img_paths(img_input_dir, num_imgs):

    img_paths = []
    for img_dir in os.scandir(img_input_dir):
        if os.path.isfile(img_dir): img_paths.append(img_dir.path)
        if (len(img_paths) >= num_imgs): break
        if os.path.isdir(img_dir.path):
            for img in os.scandir(img_dir.path):
                img_paths.append(img.path)
                if (len(img_paths) >= num_imgs): break
    return img_paths


def test_paths(attack_record_filename, result_counter_filename, output_dir):

    if (os.path.exists(attack_record_filename)): 
        sys.exit('Output file ' + attack_record_filename + ' already exists, exiting')
    if (os.path.exists(result_counter_filename)): 
        sys.exit('Output file ' + result_counter_filename + ' already exists, exiting')
    if (os.path.exists(output_dir) and len(os.listdir(output_dir)) != 0):
        user_input = input('Output directory not empty: ' + output_dir + '\nProceed? [y/N]')
        if (user_input != 'y'): sys.exit('Exiting')

def test_image(img_path, detector_dict, detector_name, tunnel_dict):
    def quality_error(error_text, counter_name, tunnel_dict):
        tunnel_dict[counter_name] += 1
        raise ValueError(error_text)

    try:
        image_labels = imagelabels.ImageLabels(img_path)
    except IOError:
        error_str = 'File Path does not exist or cannot be found: ' + str(img_path)
        quality_error(error_str, 'bad_image', tunnel_dict)
    except:
        error_str = 'Problem during init of ImageLabels (corrupted image?): ' + str(img_path)
        quality_error(error_str, 'bad_image', tunnel_dict)
    detector_dict_small = {detector_name: detector_dict[detector_name]}
    image_labels.add_detector_labels(detector_dict_small)
    if (len(image_labels.found_box_dict) > 1):
        quality_error('Found >1 true faces, skipping','multiple_faces_count',tunnel_dict)
    if (len(image_labels.found_box_dict) == 0):
        quality_error('Found 0 true faces, skipping','zero_faces_count',tunnel_dict)

    if (len(image_labels.found_box_dict[detector_name]) == 0):
        error_str = 'Could not find face prior to noise application, skipping'
        quality_error(error_str,'no_found_faces_count',tunnel_dict)

    return image_labels


