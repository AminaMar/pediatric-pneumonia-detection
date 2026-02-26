
# ═══════════════════════════════════════════════════════════
# DATA LOADING PACKAGE - PNEUMONIA DETECTION
# Data Engineer: Bouhmidi Amina Meroua
# ═══════════════════════════════════════════════════════════

import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class PneumoniaDataLoader:
    """
    Data loader for the Pneumonia Detection project
    
    Usage:
        loader = PneumoniaDataLoader(base_path)
        train_gen, val_gen, test_gen, class_weights = loader.get_generators()
        
        model.fit(
            train_gen,
            validation_data=val_gen,
            class_weight=class_weights,
            ...
        )
    """
    
    def __init__(self, base_path, img_size=224, batch_size=32, seed=42):
        self.base_path = base_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.seed = seed
        
        # Load configuration
        config_path = os.path.join(
            os.path.dirname(base_path), 
            'preprocessing_config.json'
        )
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Extract class weights
        self.class_weights = {
            int(k): v for k, v in self.config['class_weights'].items()
        }
    
    def get_generators(self):
        """
        Returns data generators for train, val, test
        
        Returns:
            train_generator, val_generator, test_generator, class_weights
        """
        # Training generator (with augmentation)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation/Test generator (no augmentation)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create the flows
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.base_path, 'train'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='binary',
            color_mode='grayscale',
            shuffle=True,
            seed=self.seed
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.base_path, 'val'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='binary',
            color_mode='grayscale',
            shuffle=False,
            seed=self.seed
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.base_path, 'test'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='binary',
            color_mode='grayscale',
            shuffle=False,
            seed=self.seed
        )
        
        return train_generator, val_generator, test_generator, self.class_weights


# ═══════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Initialize the loader
    loader = PneumoniaDataLoader(
        base_path='/content/drive/MyDrive/Pneumonia_Project/data/preprocessed',
        img_size=224,
        batch_size=32
    )
    
    # Get generators
    train_gen, val_gen, test_gen, class_weights = loader.get_generators()
    
    print("✅ Data loaders created")
    print(f"📊 Training samples: {train_gen.samples}")
    print(f"📊 Validation samples: {val_gen.samples}")
    print(f"📊 Test samples: {test_gen.samples}")
    print(f"⚖️  Class weights: {class_weights}")
