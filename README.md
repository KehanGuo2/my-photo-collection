# My Photo Collection Website 📸

A beautiful, modern, and responsive photo gallery website for showcasing your personal photo collection. **Visitor-only design** - viewers can browse, filter, and enjoy your photos without upload permissions.

## ✨ Features

- **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- **Photo Categories**: Organize photos by Landscape, Portrait, Nature, and Urban categories
- **Interactive Filtering**: Filter photos by category with smooth animations
- **Photo Sorting**: Sort photos by upload date (newest first or oldest first)
- **Upload Timestamps**: Display when each photo was added to your collection
- **Lightbox Gallery**: View photos in full-screen with navigation controls and upload dates
- **Visitor-Safe**: No upload or delete capabilities for public viewers
- **Copyright Protection**: Professional footer with copyright and contact information
- **Keyboard Navigation**: Navigate through photos using keyboard shortcuts
- **Modern UI**: Elegant black and white theme with sophisticated grey backgrounds
- **Lazy Loading**: Optimized performance with lazy loading images

## 🚀 Quick Start

### For Local Testing
1. **Download** or clone this repository to your computer
2. **Run** the local server:
   ```bash
   python server.py 8000
   ```
3. **Open** your browser to `http://localhost:8000`

### For Publishing Online
1. **Follow** the [Publishing Guide](PUBLISH.md) for detailed instructions
2. **Choose** a hosting service (GitHub Pages, Netlify, Vercel, etc.)
3. **Upload** your files and share your gallery with the world!

## 📁 File Structure

```
photo_web/
├── index.html          # Main website page
├── styles.css          # Beautiful styling
├── script.js           # Interactive functionality  
├── server.py           # Local development server
├── README.md           # This file
├── PUBLISH.md          # Publishing guide
└── images/             # Directory for your photos
```

## 🎯 How Visitors Use Your Gallery

### Viewing Photos
- **Browse the gallery** with an elegant grid layout
- **Click any photo** to open it in full-screen lightbox mode
- **Navigate** using arrow buttons or keyboard arrow keys
- **View upload dates** and photo information in overlays
- **Close** lightbox by clicking X or pressing Escape

### Filtering and Sorting Photos
- **Filter by category**: All, Landscape, Portrait, Nature, Urban
- **Sort by date**: Newest first (default) or Oldest first
- **Dynamic photo count**: See how many photos match current filter
- **Seamless experience**: Filtering and sorting work together smoothly

### Keyboard Shortcuts
- **Arrow Keys**: Navigate through photos in lightbox
- **Escape**: Close lightbox
- **Click outside**: Close lightbox

### Contact & Copyright Information
The footer provides visitors with:
- **Your contact details**: Email, phone, location
- **Social media links**: Professional networking
- **Copyright notice**: Protection for your work
- **About section**: Information about your photography

## 📸 Adding Your Own Photos

### Method 1: Replace Sample Photos
1. **Edit** `index.html` directly
2. **Replace** sample photo URLs with your own
3. **Update** titles, categories, and dates

### Method 2: Add Photo Files
1. **Create** an `images/` folder
2. **Upload** your photos there
3. **Update** HTML to reference your files:
   ```html
   <img src="images/your-photo.jpg" alt="Your Photo">
   ```

### Example Photo Entry
```html
<div class="photo-item" data-category="landscape" data-upload-date="2024-01-15T10:30:00">
    <img src="images/mountain-sunset.jpg" alt="Mountain Sunset" loading="lazy">
    <div class="photo-overlay">
        <h3>Mountain Sunset</h3>
        <p>Landscape</p>
        <p class="upload-date">Uploaded: Jan 15, 2024</p>
    </div>
</div>
```

## 🎨 Customization

### Changing Colors
Edit the CSS variables in `styles.css`:
```css
/* Main background and text colors */
background: #f5f5f5; /* Light grey background */
color: #2c2c2c; /* Dark grey text */

/* Accent colors */
color: #1a1a1a; /* Black accents */
```

### Adding New Categories
1. **Add filter button** in `index.html`:
   ```html
   <button class="filter-btn" data-filter="yourcategory">Your Category</button>
   ```
2. **Use the category** in photo items:
   ```html
   <div class="photo-item" data-category="yourcategory">
   ```

### Contact Information
Update the footer in `index.html`:
```html
<p><i class="fas fa-envelope"></i> your-email@domain.com</p>
<p><i class="fas fa-phone"></i> +1 (555) YOUR-NUMBER</p>
<p><i class="fas fa-map-marker-alt"></i> Your City, State</p>
```

## 🚀 Publishing Your Gallery

See the detailed [Publishing Guide](PUBLISH.md) for step-by-step instructions on:
- **Free hosting options** (GitHub Pages, Netlify, Vercel)
- **Paid hosting services** for custom domains
- **SEO optimization** for better visibility
- **Performance tips** for faster loading
- **Security considerations** for public galleries

## 🔧 Technical Features

- **Intersection Observer**: For efficient lazy loading
- **Date Management**: Automatic timestamp generation and formatting
- **Dynamic Sorting**: Real-time photo sorting by upload date with DOM manipulation
- **CSS Grid & Flexbox**: For responsive layouts
- **CSS Custom Properties**: For easy theming
- **ES6+ JavaScript**: Modern JavaScript features
- **Modal Management**: Advanced lightbox system with proper focus handling
- **Responsive Footer**: Professional contact and copyright information layout

## 🌐 Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge
- Mobile browsers

## 📱 Mobile Optimized

The website is fully responsive and optimized for mobile devices with:
- Touch-friendly interface
- Responsive grid layout
- Mobile-specific navigation
- Optimized image loading

## 🎯 Performance

- Lazy loading images for faster page loads
- Optimized CSS animations
- Efficient JavaScript event handling
- Responsive image sizing
- Clean, semantic HTML structure

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to customize and improve this photo gallery for your needs!

---

**Ready to showcase your photography to the world! 📷✨**

For publishing help, see: [PUBLISH.md](PUBLISH.md)
