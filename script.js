// Photo data array to store all photos
let photos = [];
let currentPhotoIndex = 0;
let filteredPhotos = [];
let currentSort = 'newest'; // Default sort order

// Initialize the website
document.addEventListener('DOMContentLoaded', function() {
    loadPhotosFromJSON();
    setupEventListeners();
    setupFilterButtons();
    setupSortButtons();
    initializeFooter();
});

// Load photos from JSON file
async function loadPhotosFromJSON() {
    try {
        console.log('🔍 Loading photos from JSON...');
        const response = await fetch('photos.json');
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const photoData = await response.json();
        
        if (!photoData.photos || photoData.photos.length === 0) {
            showNoPhotosMessage();
            return;
        }
        
        console.log(`📸 Loaded ${photoData.photos.length} photos`);
        
        // Convert JSON data to our photo format
        photos = photoData.photos.map((photo, index) => ({
            id: index,
            src: photo.path,
            title: photo.title,
            category: photo.category,
            uploadDate: new Date(photo.upload_date),
            uploadDateFormatted: photo.upload_date_formatted,
            filename: photo.filename,
            alt: photo.alt,
            size: photo.size
        }));
        
        // Sort by newest first initially
        sortPhotos('newest');
        filteredPhotos = [...photos];
        
        // Generate photo grid
        generatePhotoGrid();
        updatePhotoCount();
        
        console.log('✅ Photo gallery loaded successfully!');
        
    } catch (error) {
        console.error('❌ Error loading photos:', error);
        showErrorMessage(error.message);
    }
}

// Generate photo grid HTML dynamically
function generatePhotoGrid() {
    const photoGrid = document.getElementById('photoGrid');
    
    // Clear existing content
    photoGrid.innerHTML = '';
    
    // Create photo items
    photos.forEach((photo, index) => {
        const photoItem = createPhotoElement(photo, index);
        photoGrid.appendChild(photoItem);
        
        // Store reference to element
        photo.element = photoItem;
    });
    
    // Setup click listeners
    setupPhotoClickListeners();
    
    // Setup lazy loading
    setupLazyLoading();
}

// Create individual photo element
function createPhotoElement(photo, index) {
    const photoItem = document.createElement('div');
    photoItem.className = 'photo-item';
    photoItem.setAttribute('data-category', photo.category);
    photoItem.setAttribute('data-upload-date', photo.uploadDate.toISOString());
    
    photoItem.innerHTML = `
        <img src="${photo.src}" alt="${photo.alt}" loading="lazy">
        <div class="photo-overlay">
            <h3>${photo.title}</h3>
            <p>${photo.category.charAt(0).toUpperCase() + photo.category.slice(1)}</p>
            <p class="upload-date">Uploaded: ${photo.uploadDateFormatted}</p>
        </div>
    `;
    
    return photoItem;
}

// Show message when no photos are found
function showNoPhotosMessage() {
    const photoGrid = document.getElementById('photoGrid');
    photoGrid.innerHTML = `
        <div class="no-photos-message" style="
            grid-column: 1 / -1;
            text-align: center;
            padding: 3rem;
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        ">
            <h3 style="color: #666; margin-bottom: 1rem;">📸 No Photos Found</h3>
            <p style="color: #888; margin-bottom: 1rem;">
                Add some photos to your category folders and run the photo generator to see them here.
            </p>
            <p style="color: #999; font-size: 0.9rem;">
                Run: <code style="background: #f0f0f0; padding: 2px 6px; border-radius: 4px;">python generate_photos.py</code>
            </p>
        </div>
    `;
    updatePhotoCount();
}

// Show error message
function showErrorMessage(errorMessage) {
    const photoGrid = document.getElementById('photoGrid');
    photoGrid.innerHTML = `
        <div class="error-message" style="
            grid-column: 1 / -1;
            text-align: center;
            padding: 3rem;
            background: #fff5f5;
            border: 1px solid #fed7d7;
            border-radius: 8px;
            color: #c53030;
        ">
            <h3 style="margin-bottom: 1rem;">⚠️ Unable to Load Photos</h3>
            <p style="margin-bottom: 1rem;">${errorMessage}</p>
            <p style="font-size: 0.9rem; color: #666;">
                Make sure you've run: <code style="background: #f0f0f0; padding: 2px 6px; border-radius: 4px;">python generate_photos.py</code>
            </p>
        </div>
    `;
    updatePhotoCount();
}

// Setup event listeners for photos
function setupPhotoClickListeners() {
    photos.forEach((photo, index) => {
        if (photo.element) {
            photo.element.addEventListener('click', (e) => {
                currentPhotoIndex = filteredPhotos.findIndex(p => p.id === photo.id);
                openLightbox(photo);
            });
        }
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
    const initialSortBtn = document.querySelector(`[data-sort="${currentSort}"]`);
    if (initialSortBtn) {
        initialSortBtn.classList.add('active');
    }
    
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
            const activeFilter = document.querySelector('.filter-btn.active')?.getAttribute('data-filter') || 'all';
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
        if (photo.element && photo.element.parentNode === photoGrid) {
            photoGrid.appendChild(photo.element);
        }
    });
}

// Filter photos based on category
function filterPhotos(category) {
    if (category === 'all') {
        filteredPhotos = [...photos];
        photos.forEach(photo => {
            if (photo.element) {
                photo.element.style.display = 'block';
            }
        });
    } else {
        filteredPhotos = photos.filter(photo => photo.category === category);
        photos.forEach(photo => {
            if (photo.element) {
                if (photo.category === category) {
                    photo.element.style.display = 'block';
                } else {
                    photo.element.style.display = 'none';
                }
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
    lightboxImage.alt = photo.alt;
    lightboxTitle.textContent = photo.title;
    lightboxCategory.textContent = `${photo.category.charAt(0).toUpperCase() + photo.category.slice(1)} • Uploaded ${photo.uploadDateFormatted}`;
    
    lightbox.style.display = 'flex';
    document.body.style.overflow = 'hidden';
    
    // Add animation
    setTimeout(() => {
        lightbox.style.opacity = '1';
    }, 10);
    
    // Setup close button for this lightbox session
    setupCloseButtonForLightbox();
    
    // Show mobile swipe indicator if on mobile device
    showMobileSwipeIndicator();
}

// Setup close button specifically when lightbox opens
function setupCloseButtonForLightbox() {
    const closeBtn = document.querySelector('.close-btn');
    if (!closeBtn) return;
    
    // Remove any existing event listeners by cloning
    const newCloseBtn = closeBtn.cloneNode(true);
    closeBtn.parentNode.replaceChild(newCloseBtn, closeBtn);
    
    // Simple touch/click handler that always works
    newCloseBtn.addEventListener('touchstart', function(e) {
        e.stopPropagation(); // Prevent lightbox touch events
        this.style.transform = 'scale(0.9)';
        this.style.background = 'rgba(0, 0, 0, 0.8)';
    });
    
    newCloseBtn.addEventListener('touchend', function(e) {
        e.preventDefault();
        e.stopPropagation(); // Prevent lightbox touch events
        closeLightbox();
    });
    
    newCloseBtn.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation(); // Prevent lightbox touch events
        closeLightbox();
    });
}

function closeLightbox() {
    console.log('Closing lightbox...');
    const lightbox = document.getElementById('lightbox');
    lightbox.style.opacity = '0';
    
    setTimeout(() => {
        lightbox.style.display = 'none';
        document.body.style.overflow = 'auto';
        console.log('Lightbox closed');
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
        lightboxImage.alt = photo.alt;
        lightboxTitle.textContent = photo.title;
        lightboxCategory.textContent = `${photo.category.charAt(0).toUpperCase() + photo.category.slice(1)} • Uploaded ${photo.uploadDateFormatted}`;
        lightboxImage.style.opacity = '1';
    }, 150);
}

// Setup event listeners
function setupEventListeners() {
    // Close lightbox when clicking outside
    const lightbox = document.getElementById('lightbox');
    if (lightbox) {
        lightbox.addEventListener('click', function(e) {
            if (e.target === this) {
                closeLightbox();
            }
        });
    }
    
    // Setup close button event listeners
    setupCloseButton();
    
    // Keyboard navigation
    document.addEventListener('keydown', function(e) {
        const lightbox = document.getElementById('lightbox');
        
        if (lightbox && lightbox.style.display === 'flex') {
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
    
    // Mobile touch support for lightbox
    setupMobileTouchNavigation();
}

// Setup close button with proper touch support
function setupCloseButton() {
    const closeBtn = document.querySelector('.close-btn');
    if (closeBtn) {
        // Remove existing onclick attribute
        closeBtn.removeAttribute('onclick');
        
        // Add click event listener
        closeBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            closeLightbox();
        });
        
        // Add touch event listeners for better mobile support
        closeBtn.addEventListener('touchend', function(e) {
            e.preventDefault();
            e.stopPropagation();
            closeLightbox();
        });
        
        // Add visual feedback for touch
        closeBtn.addEventListener('touchstart', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.style.transform = 'scale(0.9)';
            this.style.background = 'rgba(255, 255, 255, 0.5)';
        });
        
        closeBtn.addEventListener('touchcancel', function(e) {
            this.style.transform = 'scale(1)';
            this.style.background = 'rgba(255, 255, 255, 0.3)';
        });
    }
}

// Setup mobile touch navigation for lightbox
function setupMobileTouchNavigation() {
    const lightbox = document.getElementById('lightbox');
    if (!lightbox) return;
    
    let startX = 0;
    let startY = 0;
    let currentX = 0;
    let currentY = 0;
    let isDragging = false;
    
    // Touch start
    lightbox.addEventListener('touchstart', function(e) {
        // Only handle single touch
        if (e.touches.length !== 1) return;
        
        // Check if touch is on close button - if so, don't handle it here
        const touch = e.touches[0];
        const closeBtn = document.querySelector('.close-btn');
        if (closeBtn) {
            const closeBtnRect = closeBtn.getBoundingClientRect();
            if (touch.clientX >= closeBtnRect.left && touch.clientX <= closeBtnRect.right &&
                touch.clientY >= closeBtnRect.top && touch.clientY <= closeBtnRect.bottom) {
                return; // Let close button handle this touch
            }
        }
        
        startX = touch.clientX;
        startY = touch.clientY;
        currentX = touch.clientX;
        currentY = touch.clientY;
        isDragging = true;
        
        // Only prevent default for swipe area (not close button area)
        e.preventDefault();
    }, { passive: false });
    
    // Touch move
    lightbox.addEventListener('touchmove', function(e) {
        if (!isDragging || e.touches.length !== 1) return;
        
        const touch = e.touches[0];
        currentX = touch.clientX;
        currentY = touch.clientY;
        
        // Calculate distance moved
        const deltaX = currentX - startX;
        const deltaY = currentY - startY;
        
        // If moving more horizontally than vertically, prevent default scrolling
        if (Math.abs(deltaX) > Math.abs(deltaY)) {
            e.preventDefault();
        }
    }, { passive: false });
    
    // Touch end
    lightbox.addEventListener('touchend', function(e) {
        if (!isDragging) return;
        
        const deltaX = currentX - startX;
        const deltaY = currentY - startY;
        const threshold = 50; // Minimum distance for swipe
        const verticalThreshold = 100; // Maximum vertical movement allowed
        
        // Check if it's a horizontal swipe
        if (Math.abs(deltaX) > threshold && Math.abs(deltaY) < verticalThreshold) {
            if (deltaX > 0) {
                // Swipe right - go to previous image
                previousImage();
            } else {
                // Swipe left - go to next image
                nextImage();
            }
        }
        
        // Reset state
        isDragging = false;
        startX = 0;
        startY = 0;
        currentX = 0;
        currentY = 0;
    });
    
    // Touch cancel
    lightbox.addEventListener('touchcancel', function(e) {
        isDragging = false;
        startX = 0;
        startY = 0;
        currentX = 0;
        currentY = 0;
    });
}

// Setup lazy loading for images
function setupLazyLoading() {
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
    const lazyImages = document.querySelectorAll('img[loading="lazy"]');
    lazyImages.forEach(img => {
        imageObserver.observe(img);
    });
}

// Refresh photo gallery (useful for testing)
function refreshGallery() {
    console.log('🔄 Refreshing photo gallery...');
    loadPhotosFromJSON();
}

// Smooth scroll to top function
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show mobile swipe indicator
function showMobileSwipeIndicator() {
    // Check if it's a mobile device
    const isMobile = window.innerWidth <= 768 || ('ontouchstart' in window);
    
    if (!isMobile || filteredPhotos.length <= 1) return;
    
    // Check if indicator already exists or has been shown
    if (document.getElementById('swipeIndicator') || localStorage.getItem('swipeIndicatorShown')) {
        return;
    }
    
    // Create swipe indicator
    const indicator = document.createElement('div');
    indicator.id = 'swipeIndicator';
    indicator.innerHTML = `
        <div class="swipe-hint">
            <i class="fas fa-hand-point-left"></i>
            <span>Swipe to navigate</span>
            <i class="fas fa-hand-point-right"></i>
        </div>
    `;
    indicator.style.cssText = `
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 12px 20px;
        border-radius: 25px;
        font-size: 0.9rem;
        z-index: 2001;
        display: flex;
        align-items: center;
        gap: 8px;
        backdrop-filter: blur(10px);
        animation: swipeIndicatorFade 3s ease-in-out forwards;
        pointer-events: none;
    `;
    
    // Add CSS animation if not already added
    if (!document.getElementById('swipeIndicatorStyles')) {
        const style = document.createElement('style');
        style.id = 'swipeIndicatorStyles';
        style.textContent = `
            @keyframes swipeIndicatorFade {
                0% { opacity: 0; transform: translateX(-50%) translateY(20px); }
                15% { opacity: 1; transform: translateX(-50%) translateY(0px); }
                85% { opacity: 1; transform: translateX(-50%) translateY(0px); }
                100% { opacity: 0; transform: translateX(-50%) translateY(-20px); }
            }
            .swipe-hint {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .swipe-hint i {
                font-size: 0.8rem;
                opacity: 0.7;
            }
        `;
        document.head.appendChild(style);
    }
    
    document.body.appendChild(indicator);
    
    // Remove indicator after animation
    setTimeout(() => {
        if (indicator && indicator.parentNode) {
            indicator.parentNode.removeChild(indicator);
        }
        // Mark as shown so it doesn't appear again
        localStorage.setItem('swipeIndicatorShown', 'true');
    }, 3000);
} 