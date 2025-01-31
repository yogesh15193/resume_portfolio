document.addEventListener("DOMContentLoaded", () => {
    console.log("✅ whyRecruitMe.js loaded");

    const cards = document.querySelectorAll("#animated-cards .card");
    let currentCardIndex = 0;
    let typingInterval;
    let isPaused = false;

    if (cards.length > 0) {
        // Initialize: Hide all cards except the first
        cards.forEach((card, index) => {
            if (index === 0) {
                card.classList.remove("hidden");
                card.style.display = "block";
            } else {
                card.classList.add("hidden");
                card.style.display = "none";
            }
        });
    } else {
        console.warn("⚠️ No cards found inside #animated-cards.");
        return; // Exit function if no cards found
    }

    // Function to type text
    function typeText(element, text, startIndex = 0, callback) {
        let i = startIndex;
        clearInterval(typingInterval);
        typingInterval = setInterval(() => {
            if (!isPaused) {
                element.textContent = text.slice(0, i + 1);
                element.parentElement.style.minHeight = `${element.scrollHeight}px`; // Adjust height dynamically
                i++;
                if (i >= text.length) {
                    clearInterval(typingInterval);
                    if (callback) callback();
                }
            }
        }, 50);
    }

    // Animate Cards
    function animateCard(index) {
        const card = cards[index];
        const textElement = card.querySelector(".typing-content");
        const text = textElement.getAttribute("data-text");

        textElement.textContent = "";
        card.classList.remove("hidden");
        card.classList.add("slide-in");

        const typingDuration = text.length * 50; // Typing speed

        typeText(textElement, text, 0, () => {
            setTimeout(() => {
                card.classList.add("slide-out");
                setTimeout(() => {
                    card.classList.add("hidden");
                    card.classList.remove("slide-in", "slide-out");
                    currentCardIndex = (currentCardIndex + 1) % cards.length;
                    animateCard(currentCardIndex);
                }, 1000);
            }, 1000 + typingDuration);
        });
    }

    // Hover effects
    cards.forEach((card) => {
        card.addEventListener("mouseover", () => {
            isPaused = true;
            clearInterval(typingInterval);
        });

        card.addEventListener("mouseout", () => {
            isPaused = false;
            const textElement = card.querySelector(".typing-content");
            const text = textElement.getAttribute("data-text");
            const alreadyTyped = textElement.textContent.length;
            if (alreadyTyped < text.length) {
                typeText(textElement, text, alreadyTyped, () => {});
            }
        });
    });

    // Start Animation
    animateCard(currentCardIndex);
});
