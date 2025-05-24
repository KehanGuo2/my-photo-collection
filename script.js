// Photo data array to store all photos
let photos = [];
let currentPhotoIndex = 0;
let filteredPhotos = [];
let currentSort = 'newest'; // Default sort order

// Initialize the website
document.addEventListener('DOMContentLoaded', function() {
    loadInitialPhotos();
    setupEventListeners();
    setupFilterButtons();
    setupSortButtons();
    initializeFooter();
    updatePhotoCount();
});

// Load initial photos from the HTML
function loadInitialPhotos() {
    const photoItems = document.querySelectorAll('.photo-item');
    photoItems.forEach((item, index) => {
        const img = item.querySelector('img');
        const title = item.querySelector('h3').textContent;
        const category = item.querySelector('p').textContent;
        const uploadDate = item.getAttribute('data-upload-date') || new Date().toISOString();
        
        photos.push({
            id: index,
            src: img.src,
            title: title,
            category: category.toLowerCase(),
            uploadDate: new Date(uploadDate),
            element: item
        });
    });
    
    // Sort by newest first initially
    sortPhotos('newest');
    filteredPhotos = [...photos];
    setupPhotoClickListeners();
}

// Setup event listeners for photos
function setupPhotoClickListeners() {
    photos.forEach((photo, index) => {
        photo.element.addEventListener('click', (e) => {
            currentPhotoIndex = filteredPhotos.findIndex(p => p.id === photo.id);
            openLightbox(photo);
        });
    });
}

// Setup filter button functionality
function setupFilterButtons() {
    const filterButtons = document.querySelectorAll('.filter-btn');
    
    filterButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons
            filterButtons.forEach(btn => btn.classList.remove('active'));
            // Add active class to clicked button
            button.classList.add('active');
            
            const filter = button.getAttribute('data-filter');
            filterPhotos(filter);
        });
    });
}

// Setup sort button functionality
function setupSortButtons() {
    const sortButtons = document.querySelectorAll('.sort-btn');
    
    // Set initial active state
    document.querySelector(`[data-sort="${currentSort}"]`).classList.add('active');
    
    sortButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all sort buttons
            sortButtons.forEach(btn => btn.classList.remove('active'));
            // Add active class to clicked button
            button.classList.add('active');
            
            const sortType = button.getAttribute('data-sort');
            currentSort = sortType;
            sortPhotos(sortType);
            
            // Re-apply current filter after sorting
            const activeFilter = document.querySelector('.filter-btn.active').getAttribute('data-filter');
            filterPhotos(activeFilter);
        });
    });
}

// Sort photos based on upload date
function sortPhotos(sortType) {
    photos.sort((a, b) => {
        if (sortType === 'newest') {
            return b.uploadDate - a.uploadDate; // Newest first
        } else {
            return a.uploadDate - b.uploadDate; // Oldest first
        }
    });
    
    // Re-arrange DOM elements
    const photoGrid = document.getElementById('photoGrid');
    photos.forEach(photo => {
        photoGrid.appendChild(photo.element);
    });
}

// Filter photos based on category
function filterPhotos(category) {
    const photoItems = document.querySelectorAll('.photo-item');
    
    if (category === 'all') {
        filteredPhotos = [...photos];
        photoItems.forEach(item => {
            item.style.display = 'block';
        });
    } else {
        filteredPhotos = photos.filter(photo => photo.category === category);
        photoItems.forEach(item => {
            const photoCategory = item.getAttribute('data-category');
            if (photoCategory === category) {
                item.style.display = 'block';
            } else {
                item.style.display = 'none';
            }
        });
    }
    
    updatePhotoCount();
}

// Update photo count in header
function updatePhotoCount() {
    const photoCountElement = document.getElementById('photoCount');
    if (photoCountElement) {
        photoCountElement.textContent = filteredPhotos.length;
    }
}

// Format date for display
function formatDate(date) {
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

// Initialize footer with current date
function initializeFooter() {
    const lastUpdatedElement = document.getElementById('lastUpdated');
    if (lastUpdatedElement) {
        lastUpdatedElement.textContent = formatDate(new Date());
    }
}

// Lightbox functionality
function openLightbox(photo) {
    const lightbox = document.getElementById('lightbox');
    const lightboxImage = document.getElementById('lightboxImage');
    const lightboxTitle = document.getElementById('lightboxTitle');
    const lightboxCategory = document.getElementById('lightboxCategory');
    
    lightboxImage.src = photo.src;
    lightboxImage.alt = photo.title;
    lightboxTitle.textContent = photo.title;
    lightboxCategory.textContent = `${photo.category.charAt(0).toUpperCase() + photo.category.slice(1)} • Uploaded ${formatDate(photo.uploadDate)}`;
    
    lightbox.style.display = 'flex';
    document.body.style.overflow = 'hidden';
    
    // Add animation
    setTimeout(() => {
        lightbox.style.opacity = '1';
    }, 10);
}

function closeLightbox() {
    const lightbox = document.getElementById('lightbox');
    lightbox.style.opacity = '0';
    
    setTimeout(() => {
        lightbox.style.display = 'none';
        document.body.style.overflow = 'auto';
    }, 300);
}

function previousImage() {
    if (filteredPhotos.length === 0) return;
    
    currentPhotoIndex = (currentPhotoIndex - 1 + filteredPhotos.length) % filteredPhotos.length;
    const photo = filteredPhotos[currentPhotoIndex];
    updateLightboxImage(photo);
}

function nextImage() {
    if (filteredPhotos.length === 0) return;
    
    currentPhotoIndex = (currentPhotoIndex + 1) % filteredPhotos.length;
    const photo = filteredPhotos[currentPhotoIndex];
    updateLightboxImage(photo);
}

function updateLightboxImage(photo) {
    const lightboxImage = document.getElementById('lightboxImage');
    const lightboxTitle = document.getElementById('lightboxTitle');
    const lightboxCategory = document.getElementById('lightboxCategory');
    
    // Add fade effect
    lightboxImage.style.opacity = '0';
    
    setTimeout(() => {
        lightboxImage.src = photo.src;
        lightboxImage.alt = photo.title;
        lightboxTitle.textContent = photo.title;
        lightboxCategory.textContent = `${photo.category.charAt(0).toUpperCase() + photo.category.slice(1)} • Uploaded ${formatDate(photo.uploadDate)}`;
        lightboxImage.style.opacity = '1';
    }, 150);
}

// Setup event listeners
function setupEventListeners() {
    // Close lightbox when clicking outside
    document.getElementById('lightbox').addEventListener('click', function(e) {
        if (e.target === this) {
            closeLightbox();
        }
    });
    
    // Keyboard navigation
    document.addEventListener('keydown', function(e) {
        const lightbox = document.getElementById('lightbox');
        
        if (lightbox.style.display === 'flex') {
            switch(e.key) {
                case 'Escape':
                    closeLightbox();
                    break;
                case 'ArrowLeft':
                    previousImage();
                    break;
                case 'ArrowRight':
                    nextImage();
                    break;
            }
        }
    });
}

// Smooth scroll to top function
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Lazy loading intersection observer
const imageObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const img = entry.target;
            img.src = img.dataset.src || img.src;
            img.classList.remove('lazy');
            observer.unobserve(img);
        }
    });
});

// Observe all images for lazy loading
document.addEventListener('DOMContentLoaded', function() {
    const lazyImages = document.querySelectorAll('img[loading="lazy"]');
    lazyImages.forEach(img => {
        imageObserver.observe(img);
    });
}); 