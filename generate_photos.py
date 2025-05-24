#!/usr/bin/env python3
"""
Photo Gallery Generator
Automatically scans photo folders and generates photo data for the website
"""

import os
import json
import datetime
from pathlib import Path

def get_photo_info(filepath, category):
    """Extract photo information from file"""
    file_path = Path(filepath)
    
    # Get file stats
    stat = file_path.stat()
    
    # Convert filename to title (remove extension, replace dashes/underscores with spaces, title case)
    title = file_path.stem.replace('-', ' ').replace('_', ' ').title()
    
    # Use file modification time as upload date
    upload_date = datetime.datetime.fromtimestamp(stat.st_mtime)
    
    # Generate relative path for web
    web_path = str(file_path).replace('\\', '/')
    
    return {
        'filename': file_path.name,
        'path': web_path,
        'title': title,
        'category': category,
        'upload_date': upload_date.isoformat(),
        'upload_date_formatted': upload_date.strftime("%b %d, %Y"),
        'size': stat.st_size,
        'alt': f"{title} - {category.title()} Photography"
    }

def scan_photo_folders():
    """Scan all photo category folders and collect photo data"""
    
    # Define photo categories and their folders
    categories = {
        'landscape': 'photos/landscape',
        'portrait': 'photos/portrait', 
        'nature': 'photos/nature',
        'urban': 'photos/urban'
    }
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
    
    all_photos = []
    
    for category, folder_path in categories.items():
        if not os.path.exists(folder_path):
            print(f"📁 Creating folder: {folder_path}")
            os.makedirs(folder_path, exist_ok=True)
            continue
            
        folder = Path(folder_path)
        photo_count = 0
        
        # Scan for image files
        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                try:
                    photo_info = get_photo_info(file_path, category)
                    all_photos.append(photo_info)
                    photo_count += 1
                    print(f"📸 Found: {file_path.name} ({category})")
                except Exception as e:
                    print(f"⚠️  Error processing {file_path.name}: {e}")
        
        print(f"✅ {category.title()}: {photo_count} photos")
    
    # Sort photos by upload date (newest first)
    all_photos.sort(key=lambda x: x['upload_date'], reverse=True)
    
    return all_photos

def generate_photos_json():
    """Generate photos.json file with all photo data"""
    
    print("🔍 Scanning photo folders...")
    photos = scan_photo_folders()
    
    # Create photo data structure
    photo_data = {
        'generated_at': datetime.datetime.now().isoformat(),
        'total_photos': len(photos),
        'categories': {},
        'photos': photos
    }
    
    # Count photos by category
    for photo in photos:
        category = photo['category']
        if category not in photo_data['categories']:
            photo_data['categories'][category] = 0
        photo_data['categories'][category] += 1
    
    # Write JSON file
    output_file = 'photos.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(photo_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Generated {output_file} with {len(photos)} photos!")
    print(f"📊 Photo breakdown:")
    for category, count in photo_data['categories'].items():
        print(f"   • {category.title()}: {count} photos")
    
    return photo_data

def update_gitignore():
    """Add photos.json to gitignore if not already there"""
    gitignore_path = '.gitignore'
    gitignore_content = ''
    
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()
    
    if 'photos.json' not in gitignore_content:
        with open(gitignore_path, 'a') as f:
            f.write('\n# Auto-generated photo data\nphotos.json\n')
        print("📝 Added photos.json to .gitignore")

if __name__ == "__main__":
    print("📸 Photo Gallery Generator")
    print("=" * 40)
    
    try:
        # Generate the photos data
        photo_data = generate_photos_json()
        
        # Update gitignore
        update_gitignore()
        
        print("\n✨ Photo gallery updated successfully!")
        print("🌐 Refresh your website to see the new photos.")
        
        if photo_data['total_photos'] == 0:
            print("\n💡 Tip: Add some photos to the folders in photos/ and run this script again.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1) 