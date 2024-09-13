from fastai import *
from fastbook import *
from PIL import Image    
import time
from pathlib import Path
from fastai.vision.all import parallel
import hashlib
import random
import os
import random
import shutil

download_other_images = False
download_cats_images = False
perform_train_test_split = True

TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.1
VALID_SPLIT = 0.1

# image download must be less than 1000 each time
IMAGE_DOWNLOAD_COUNT = 900
REQUIRED_IMAGES = 800
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

IMAGE_RESIZE = (460, 460)
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100

path = Path('inputs')
raw_path = Path(path/'raw')
cat_path = raw_path/'cats'
not_a_pet_path = Path(raw_path/"not-a-pet")
processed_path = Path(path/'processed')

cat_breeds = [
    "Abyssinian",
    "American Bobtail",
    "American Curl",
    "American Shorthair",
    "American Wirehair",
    "Balinese",
    "Bengal",
    "Birman",
    "Bombay",
    "British Shorthair",
    "Burmese",
    "Chartreux",
    "Ragdoll",
    "Exotic Shorthair",
    "Persian",
    "Maine Coon",
    "Sphynx",
    "Siamese",
    "Turkish Van",
    "Scottish Fold",
    "Devon Rex",
    "Cornish Rex",
    "Norwegian Forest Cat",
    "Russian Blue",
    "Egyptian Mau",
    "Oriental Shorthair",
    "Japanese Bobtail",
    "Somali",
    "Manx",
    "Singapura",
    "LaPerm",
    "Turkish Angora",
    "Ocicat",
    "Tonkinese",
    "Havana Brown",
    "Siberian",
    "Snowshoe",
    "Selkirk Rex",
    "Savannah",
    "Khao Manee",
    "Lykoi",
    "Toyger",
    "Peterbald",
    "Munchkin",
    "Cheetoh"
]

other_type = [
    'human faces',
    'trees',
    'forests',
    'buildings', 
    'sea',
    'boats',
    'cars'
]

def resize_and_save(img_path, dest_path, size=IMAGE_RESIZE):
    try:
        with Image.open(img_path) as img:
            # Convert image to RGB if necessary
            if img.mode in ("RGBA", "P"):
                print(f'Converting {img_path} to RGB')
                img = img.convert("RGB")

            # if image size is less than MIN_IMAGE_WIDTH*MIN_IMAGE_HEIGHT, skip
            if img.width < MIN_IMAGE_WIDTH or img.height < MIN_IMAGE_HEIGHT:
                return
            
            # Resize the image (example: resize to 256x256)
            img.resize(size, Image.LANCZOS)
            
            # Save the image to the target folder
            img.save(dest_path, format='JPEG')        
    except Exception as e:
        print(f"Error resizing {img_path}: {e}")

def resize_and_save_parallel(args):
    img_path, target_folder = args
    dest_path = target_folder / img_path.name
    resize_and_save(img_path, dest_path)
    
def calculate_md5(file_path):
    """Calculate the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def remove_duplicate_images(folder_path):
    """Remove duplicate images based on MD5 hash."""
    
    print("Removing duplicate images...")
    image_files = get_image_files(folder_path)
    
    seen_hashes = set()
    for img_path in image_files:
        img_hash = calculate_md5(img_path)
        if img_hash in seen_hashes:
            os.remove(img_path)
            print(f"Removed duplicate image: {img_path}")
        else:
            seen_hashes.add(img_hash)

def remove_invalid_files_not_jpg_jpeg(folder_path):
    """Remvoe invalid files not jpg or jpeg.

    Args:
        folder_path (_type_): _description_
    """
    # Gather all image files recursively
    all_images = get_image_files(folder_path, recurse=True)
    
    # Regular expression to match valid .jpg or .jpeg files even with extra characters after
    pattern = re.compile(r'\.(jpe?g)$', re.IGNORECASE)
    # delete images not jpg or jpeg
    for img in all_images:
        if not pattern.search(str(img)):
            print(f"Deleting invalid image format: {img}")  # Optional: print for debugging
            img.unlink()  # Delete the file

def verify_images_are_valid(folder_path):
    all_images = get_image_files(folder_path)
 
    # Verify images
    print("Verifying images...")
    failed = verify_images(all_images)
    print(f"Failed images: {failed}")

    # Unlink (delete) failed images
    if len(failed) > 0:
        print("Deleting failed images...")
        for img in failed:
            try:    
                print(f"Deleting failed image: {img}")
                img.unlink()
            except Exception as e:
                print(f"Error deleting {img}: {e}")

if download_other_images:
    for other in other_type:
        print(f"Preparing to download {other} images to {not_a_pet_path}")
        results = search_images_ddg(other)
        download_images(not_a_pet_path, urls=results)

if download_cats_images:
    for cat in sorted(cat_breeds):
        dest = Path(cat_path/cat)
        print(f"Preparing to download {cat} images to {dest}")
        
        # Check if the destination folder already has more than required image count
        if dest.exists() and len(list(dest.glob('*'))) > REQUIRED_IMAGES:
            print(f"Skipping {cat} as it already has more than {REQUIRED_IMAGES} images.")
            continue
        
        retries = 0
        while retries < MAX_RETRIES:
            try:
                print(f"Downloading images for {cat}...")
                results = search_images_ddg(f'{cat} cat')
                download_images(dest, urls=results, max_pics=900)
                break  # Exit loop if successful
            except Exception as e:
                retries += 1
                print(f"Error downloading images for {cat}: {e}")
                if retries < MAX_RETRIES:
                    print(f"Retrying... ({retries}/{MAX_RETRIES})")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"Failed to download images for {cat} after {MAX_RETRIES} attempts.")
   
if perform_train_test_split:
    remove_invalid_files_not_jpg_jpeg(path)

    verify_images_are_valid(path)
    
    # all_images = get_image_files(path)
    remove_duplicate_images(path)

    # perform test / train / valid split
    train_path = Path(processed_path/'train')
    test_path = Path(processed_path/'test')
    valid_path = Path(processed_path/'valid')

    # delete train and test directories if they exist
    if train_path.exists():
        shutil.rmtree(train_path)

    if test_path.exists():
        shutil.rmtree(test_path)

    if valid_path.exists():
        shutil.rmtree(valid_path)
    
    # create processed_path if it does not exist
    if not processed_path.exists():
        processed_path.mkdir()
        
    train_path.mkdir()
    test_path.mkdir()
    valid_path.mkdir()
        
    # Initialize an empty list to hold all tasks
    all_tasks = []

    # Get all folders recursively
    all_folders = [f for f in raw_path.rglob('*') if f.is_dir()]

    # Filter out parent directories that have subdirectories
    folders = [f for f in all_folders if not any(sub.is_dir() for sub in f.iterdir())]

    # Iterate over each folder and create tasks for each image
    for folder in folders:
        # Get a list of images in the folder
        images = [img for ext in ('*.jpg', '*.jpeg') for img in folder.rglob(ext)]
        print(f"Found {len(images)} images in {folder}")
        
        # Shuffle the images
        random.shuffle(images)
        # Calculate the split indices
        train_split_idx = int(TRAIN_SPLIT * len(images))
        test_split_idx  = int((TRAIN_SPLIT + TEST_SPLIT) * len(images))
        
        # Create tasks for each image
        for i, img in enumerate(images):
            if i < train_split_idx:
                target_dir = train_path
            elif i < test_split_idx:
                target_dir = test_path
            else:
                target_dir = valid_path
            
            # move cats into cats folder
            # not-a-pet into not-a-pet folder
            if folder.parent.name == 'cats':
                target_dir = target_dir / 'cats'                  
            target_folder = target_dir / folder.name

            if not target_folder.exists():
                target_folder.mkdir(parents=True, exist_ok=True)
            all_tasks.append((img, target_folder))

    # Use FastAI's parallel function to resize and save images in parallel
    n_workers = os.cpu_count()  # Get the number of CPU cores
    print(f"Resizing and saving {len(all_tasks)} images in parallel using {n_workers} workers...")
    parallel(resize_and_save_parallel, all_tasks, n_workers=n_workers)