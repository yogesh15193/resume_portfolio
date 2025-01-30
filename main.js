// import './style.css'
// import javascriptLogo from './javascript.svg'
// import viteLogo from '/vite.svg'
// import { setupCounter } from './counter.js'

console.log("Hello World")

document.addEventListener('DOMContentLoaded', function() {
    // Get all buttons that should trigger the popup
    const buttons = document.querySelectorAll('button');
    
    buttons.forEach(button => {
        const buttonText = button.textContent.toLowerCase();
        if (buttonText.includes('get details') || 
            buttonText.includes('enquire') || 
            buttonText.includes('download brochure') ||
            buttonText.includes('schedule a visit') ||
            buttonText.includes('contact')) {
            
            // Randomly assign different animation classes
            const animations = [
                'animate-pulse-custom',
                'animate-bounce-custom',
                'animate-shake',
                'animate-scale',
                'animate-glow',
                'btn-animated'
            ];
            
            // Add base classes for all buttons
            button.classList.add('transition-all', 'duration-300');
            
            // Add specific animation class
            // You can either randomly assign animations:
            
            
            // Or assign specific animations based on button type:
            if (buttonText.includes('enquire')) {
                button.classList.add('animate-pulse-custom');
            } else if (buttonText.includes('download')) {
                button.classList.add('animate-pulse-custom');
            } else if (buttonText.includes('schedule')) {
                button.classList.add('animate-pulse-custom');
            } else {
                button.classList.add('animate-pulse-custom');
            }
            
            // Add hover effect for all buttons
            button.addEventListener('mouseenter', () => {
                button.style.transform = 'translateY(-2px)';
            });
            
            button.addEventListener('mouseleave', () => {
                button.style.transform = 'translateY(0)';
            });
        }
    });
});


function reveal() {
    const reveals = document.querySelectorAll('.reveal');
    
    reveals.forEach(element => {
        const windowHeight = window.innerHeight;
        const elementTop = element.getBoundingClientRect().top;
        const elementVisible = 150;
        
        if (elementTop < windowHeight - elementVisible) {
            element.classList.add('active');
        }
    });
}

window.addEventListener('scroll', reveal);
reveal(); // Initial check


const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');

    mobileMenuButton.addEventListener('click', () => {
        mobileMenu.classList.toggle('hidden');
    });
    function toggleRows(targetId, button) {
        const hiddenRows = document.getElementById(targetId);
        const buttonText = button.querySelector('span');
        const icon = button.querySelector('svg');
        
        if (hiddenRows.classList.contains('hidden')) {
            hiddenRows.classList.remove('hidden');
            buttonText.textContent = 'Show Less';
            icon.style.transform = 'rotate(180deg)';
        } else {
            hiddenRows.classList.add('hidden');
            buttonText.textContent = 'Show More';
            icon.style.transform = 'rotate(0deg)';
        }
    }

    function openEnquiryPopup() {
        document.getElementById('enquiryPopup').classList.remove('hidden');
        document.body.style.overflow = 'hidden'; // Prevent background scrolling
    }

    // Function to close enquiry popup
    function closeEnquiryPopup() {
        document.getElementById('enquiryPopup').classList.add('hidden');
        document.body.style.overflow = 'auto'; // Restore scrolling
    }

    // Handle form submission
    function handleEnquirySubmit(event) {
        event.preventDefault();
        // Add your form submission logic here
        alert('Thank you for your enquiry. We will get back to you soon!');
        closeEnquiryPopup();
    }

    // Add click event listeners to all "Get Details" buttons
    document.addEventListener('DOMContentLoaded', function() {
        // Get all buttons that should trigger the popup
        const buttons = document.querySelectorAll('button');
        
        buttons.forEach(button => {
            // Check if the button text contains specific keywords
            const buttonText = button.textContent.toLowerCase();
            if (buttonText.includes('get details') || 
                buttonText.includes('enquire') || 
                buttonText.includes('download brochure') ||
                buttonText.includes('schedule a visit') ||
                buttonText.includes('contact')) {
                button.addEventListener('click', openEnquiryPopup);
            }
        });

        // Close popup when clicking outside
        document.getElementById('enquiryPopup').addEventListener('click', function(e) {
            if (e.target === this) {
                closeEnquiryPopup();
            }
        });

        // Close popup with Escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeEnquiryPopup();
            }
        });
    });

    document.querySelectorAll('button').forEach(button => {
        if (button.textContent.trim() === 'Enquire Now') {
            button.addEventListener('click', () => {
                document.getElementById('enquiryPopup').classList.remove('hidden');
            });
        }
    });

    document.getElementById('closePopup').addEventListener('click', () => {
        document.getElementById('enquiryPopup').classList.add('hidden');
    });

    
// document.querySelector('#app').innerHTML = `
//   <div>
//     <a href="https://vitejs.dev" target="_blank">
//       <img src="${viteLogo}" class="logo" alt="Vite logo" />
//     </a>
//     <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript" target="_blank">
//       <img src="${javascriptLogo}" class="logo vanilla" alt="JavaScript logo" />
//     </a>
//     <h1>Hello Vite!</h1>
//     <div class="card">
//       <button id="counter" type="button"></button>
//     </div>
//     <p class="read-the-docs">
//       Click on the Vite logo to learn more
//     </p>
//   </div>
// `

// setupCounter(document.querySelector('#counter'))
