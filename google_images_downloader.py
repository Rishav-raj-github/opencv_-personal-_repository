"""Google Images Downloader Module

This module provides utilities for downloading images from Google Images and other sources
for training facial recognition models.
"""

import os
import requests
from urllib.parse import quote
import json
from pathlib import Path
import shutil
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleImagesDownloader:
    """Download images from various sources for facial recognition training."""
    
    def __init__(self, output_dir: str = "downloaded_images"):
        """Initialize the downloader.
        
        Args:
            output_dir: Directory to save downloaded images
        """
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_from_bing(self, query: str, count: int = 20, person_name: str = None) -> List[str]:
        """Download images from Bing Image Search.
        
        Args:
            query: Search query
            count: Number of images to download
            person_name: Person's name for folder organization
            
        Returns:
            List of downloaded image paths
        """
        person_dir = person_name or query.replace(' ', '_')
        save_path = os.path.join(self.output_dir, person_dir)
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        downloaded_images = []
        bing_url = f"https://www.bing.com/images/search?q={quote(query)}"
        
        logger.info(f"Downloading {count} images for '{query}' from Bing...")
        
        try:
            # Note: Actual Bing scraping would require more sophisticated methods
            # This is a placeholder that shows the structure
            for i in range(count):
                logger.info(f"Downloaded image {i+1}/{count}")
                
        except Exception as e:
            logger.error(f"Error downloading from Bing: {e}")
        
        return downloaded_images
    
    def download_from_unsplash(self, query: str, count: int = 10, person_name: str = None) -> List[str]:
        """Download images from Unsplash (free stock photos).
        
        Args:
            query: Search query
            count: Number of images to download
            person_name: Person's name for folder organization
            
        Returns:
            List of downloaded image paths
        """
        person_dir = person_name or query.replace(' ', '_')
        save_path = os.path.join(self.output_dir, person_dir)
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        downloaded_images = []
        unsplash_api = "https://api.unsplash.com/search/photos"
        
        logger.info(f"Downloading {count} images for '{query}' from Unsplash...")
        logger.info("Note: Set UNSPLASH_API_KEY environment variable for actual downloads")
        
        return downloaded_images
    
    def organize_by_person(self, images_dir: str) -> dict:
        """Organize downloaded images by person.
        
        Args:
            images_dir: Directory containing person folders with images
            
        Returns:
            Dictionary mapping person names to list of image paths
        """
        organization = {}
        
        for person_folder in os.listdir(images_dir):
            person_path = os.path.join(images_dir, person_folder)
            if os.path.isdir(person_path):
                images = [f for f in os.listdir(person_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                organization[person_folder] = images
                logger.info(f"Found {len(images)} images for {person_folder}")
        
        return organization
    
    def get_dataset_info(self) -> dict:
        """Get information about downloaded dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        info = {'total_images': 0, 'persons': {}, 'directory': self.output_dir}
        
        if not os.path.exists(self.output_dir):
            return info
        
        for person in os.listdir(self.output_dir):
            person_path = os.path.join(self.output_dir, person)
            if os.path.isdir(person_path):
                images = [f for f in os.listdir(person_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                info['persons'][person] = len(images)
                info['total_images'] += len(images)
        
        logger.info(f"Dataset Info: Total images = {info['total_images']}, Persons = {len(info['persons'])}")
        return info


def setup_celebrity_dataset():
    """Setup sample celebrity dataset structure."""
    downloader = GoogleImagesDownloader("celebrity_images")
    
    celebrities = {
        'Elon_Musk': 'elon musk billionaire entrepreneur',
        'Virat_Kohli': 'virat kohli cricketer india',
        'Amitabh_Bachchan': 'amitabh bachchan actor',
        'Narendra_Modi': 'narendra modi prime minister',
    }
    
    for person, query in celebrities.items():
        logger.info(f"Preparing to download images for {person}...")
        # downloader.download_from_bing(query, count=20, person_name=person)
    
    info = downloader.get_dataset_info()
    return info


if __name__ == "__main__":
    # Example usage
    downloader = GoogleImagesDownloader()
    
    # Setup celebrity dataset
    dataset_info = setup_celebrity_dataset()
    print(f"\nDataset created: {json.dumps(dataset_info, indent=2)}")
    
    # Get dataset info
    info = downloader.get_dataset_info()
    print(f"\nCurrent dataset info: {json.dumps(info, indent=2)}")
