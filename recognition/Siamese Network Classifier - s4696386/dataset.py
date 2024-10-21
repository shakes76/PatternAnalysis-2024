from tkinter import filedialog
import os, torch, random, torchvision

BENIGN = 0
MALIGNANT = 1
SIMILAR = 1.0
DISSIMILAR = 0.0
IMG_FILE_TYPE = ".jpg"
DEFAULT_LOCATION = "E:/COMP3710 Project/Images", "E:/COMP3710 Project/Truths.csv"
PROCESSED_DATA = tuple[dict, list[str], list[str]] # Used for typehinting

"""
AI and MT tools used to
 - Explain torchvision package details that might be useful for data augmentation
 - Explain torch Datasets vs Dataloaders and how to use them (although my understanding of Dataloaders is still... shaky)
 - Write docstrings (Removed since I'm not sure that's 'appropriate use') (I did not write any myself)
 - Explaining and interpreting errors (and suggesting possible fixes)
        (especially shape before preprocessed data became available
        https://edstem.org/au/courses/18266/discussion/2303245)
"""

# Imitating:
# https://github.com/pytorch/examples/blob/main/siamese_network
def read_data(file_path_image_folder: str = None,file_path_ground_truth: str = None
        ) -> tuple[dict[str, str], dict[str, tuple[str, int]], list[str], list[str]]:
    
    # Ensure we have a directory to take data from and a collection of ground truths
    if (not os.path.exists(file_path_image_folder) or
        not os.path.exists(file_path_ground_truth)):
        file_path_image_folder, file_path_ground_truth = get_path()
    
    # Move to that directory as our current working directory
    os.chdir(file_path_image_folder)
    
    # Maintain list of malignant & benign images
    malignants: list[str] = []
    benigns: list[str] = []

    # Populate dict and lists
    malignants, benigns = read_truths(file_path_ground_truth)
    
    # Create a mapping from image name to image
    images: dict = {}
    # Cut down the data to balance classes (and avoid a vast minority)
    benigns = benigns[:len(malignants)*2]
    new_benigns = []
    new_malignants = []

    # Load and augment images (effectively doubling)
    for b in benigns:
        b2 = b+"_2"
        image1, image2 = load_image(b+IMG_FILE_TYPE)
        images[b] = image1
        images[b2] = image2
        new_benigns.append(b2)
    benigns.extend(new_benigns)
        
    for m in malignants:
        m2 = m+"_2"
        image1, image2 = load_image(m+IMG_FILE_TYPE)
        images[m] = image1
        images[m2] = image2
        new_malignants.append(m2)
    malignants.extend(new_malignants)
    
    print(f"Loaded {len(malignants)} malignants\n")
    
    return images, malignants, benigns

def read_truths(file_path_ground_truth):
    # Read truths from ground_truth file
    malignants, benigns = [], []
    with open(file_path_ground_truth) as file_ground_truth:
        for i, line in enumerate(file_ground_truth):
            if i == 0:
                continue
            _, image_name, _, malignant = line.split(",")
            # Assign numerical values to malignance
            malignant = int(malignant)
            if malignant:
                malignants.append(image_name)
            else:
                benigns.append(image_name)
    return malignants, benigns

def get_path():
    file_path_image_folder = filedialog.askdirectory()
    file_path_ground_truth = filedialog.askopenfile().name
    return file_path_image_folder, file_path_ground_truth

def load_image(file_name):
    image = torchvision.io.read_image(file_name)
    # Define data augmentation transformations
    augment_and_resize = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(1),  # Always flip the image horizontally
        torchvision.transforms.RandomRotation(20),       # Randomly rotate the image
    ])
    image2 = augment_and_resize(image).float()
    image1 = image.float()
    return image1, image2


class Siamese_DataSet(torch.utils.data.Dataset):
    
    RANDOM_SEED = 69420

    def __init__(self, processed_data: PROCESSED_DATA,
            train: bool, train_ratio: float = 0.8):
        super(Siamese_DataSet, self).__init__()
        
        # Store splitting data
        self._train = train
        self.train_ratio = train_ratio
        self.test_ratio = 1 - train_ratio

        # Use read_data function to load the local dataset
        self.images, self.malignants, self.benigns = processed_data

        # Group the examples based on malignancy
        self.grouped_examples = {BENIGN: self.benigns, MALIGNANT: self.malignants}

        # Split the examples into training and testing
        self.train_examples, self.test_examples = self.split_dataset(self.train_ratio)
        
        # Determine which type this Dataloader is
        self.data_set = self.train_examples if self.is_train_set() else self.test_examples

    def split_dataset(self, train_ratio: float) -> tuple[dict[int, list], dict[int, list]]:
        random.seed(self.RANDOM_SEED)
        train_examples = {BENIGN: [], MALIGNANT: []}
        test_examples = {BENIGN: [], MALIGNANT: []}

        # Shuffle dataset and then split into test and train (using provided ratio)
        for malignancy in self.grouped_examples:
            examples = self.grouped_examples[malignancy]
            random.shuffle(examples)
            split_index = int(len(examples) * train_ratio)
            train_examples[malignancy] = examples[:split_index]
            test_examples[malignancy] = examples[split_index:]

        return train_examples, test_examples

    def __len__(self) -> int:
        return sum([len(val) for key, val in self.data_set.items()])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Choose a random class (BENIGN or MALIGNANT) for the first image
        selected_class = random.choice([BENIGN, MALIGNANT])

        # Select a random image from the chosen class
        img_name_1 = random.choice(self.data_set[selected_class])
        image_1 = self.images.get(img_name_1, None)

        if index % 2 == 0:
            # Positive example: Pick a different image from the same class
            img_name_2 = img_name_1
            while img_name_2 == img_name_1:
                img_name_2 = random.choice(self.data_set[selected_class])
            target = torch.tensor(SIMILAR)  # Positive label
        else:
            # Negative example: Pick an image from the other class
            other_class = MALIGNANT if selected_class == BENIGN else BENIGN
            img_name_2 = random.choice(self.data_set[other_class])
            target = torch.tensor(DISSIMILAR)  # Negative label

        image_2 = self.images.get(img_name_2, None)

        return image_1, image_2, target
    
    def is_train_set(self) -> bool:
        return self._train
    
class Classifier_DataSet(Siamese_DataSet):

    def __init__(self, processed_data: tuple[dict, list[str], list[str]], train: bool, train_ratio: float = 0.8):
        super().__init__(processed_data, train, train_ratio)

        self.malignants = self.data_set.get(MALIGNANT, None)
        self.benigns = self.data_set.get(BENIGN, None)

        if self.malignants == None or self.benigns == None:
            print("FAILED to migrate dataset correctly") # Just in case :(
        
        # Combine malignants and benigns into 1 list with shape (image_name, malignance)
        self.data_set = [(i, MALIGNANT) for i in self.malignants] + [(i, BENIGN) for i in self.benigns]
        self.length = len(self.data_set)

    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # To avoid going out of bounds (just in case) (although it shouldn't happen?)
        index = index % self.length

        image_name, target = self.data_set[index]
        return self.images.get(image_name), target

