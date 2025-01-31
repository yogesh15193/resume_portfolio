document.addEventListener("DOMContentLoaded", function () {
    const reviewSection = document.getElementById("review-container");
    const reviewHeading = document.getElementById("review-heading");
    const reviewCarousel = document.getElementById("review-carousel");
    const reviewForm = document.getElementById("recommendation-form");
    
    const OWNER_PASSCODE = "mySecret123";  // Change this to your desired password
    
    let reviews = JSON.parse(localStorage.getItem("submittedReviews")) || [];
    
    function toggleReviewSection() {
        if (reviews.length === 0) {
            reviewSection.style.display = "none";
            reviewHeading.style.display = "none";
        } else {
            reviewSection.style.display = "flex";
            reviewHeading.style.display = "block";
        }
    }
    toggleReviewSection();
    
    const urlParams = new URLSearchParams(window.location.search);
    const approvedIndex = urlParams.get("approve_deletion");
    
    if (approvedIndex !== null && approvedIndex !== "**") {  
        let passcode = prompt("Enter admin passcode to confirm deletion:");
    
        if (passcode === "mySecret123") {  
            let reviews = JSON.parse(localStorage.getItem("submittedReviews")) || [];
    
            if (approvedIndex >= 0 && approvedIndex < reviews.length) {
                reviews.splice(approvedIndex, 1); // Remove review at index
                localStorage.setItem("submittedReviews", JSON.stringify(reviews));
    
                alert("Review successfully deleted.");
    
                // 🔥 **FIX: Remove `approve_deletion` parameter to prevent re-execution**
                window.history.replaceState(null, "", window.location.pathname);
                location.reload(); // Refresh without query params
            } else {
                alert("Invalid review index. Deletion failed.");
            }
        } else {
            alert("Incorrect passcode! Review deletion not authorized.");
        }
    }
    
    function displayReviews() {
        reviewCarousel.innerHTML = "";
    
        let layoutClass = "justify-center";
        if (reviews.length === 2) layoutClass = "justify-between";
        else if (reviews.length >= 3) layoutClass = "justify-around";
    
        reviewCarousel.className = `flex ${layoutClass} space-x-8 transition-all duration-1000`;
    
        reviews.slice(-3).forEach((review, index) => {
            let reviewBox = document.createElement("div");
            reviewBox.classList.add("bg-white", "p-6", "rounded-lg", "shadow-lg", "text-center", "relative");
            reviewBox.style.minWidth = "250px";
            reviewBox.style.flex = "1";
            reviewBox.style.height = "auto";
            reviewBox.style.aspectRatio = "1 / 1";
    
            reviewBox.innerHTML = `
                <h3 class="text-lg font-semibold text-gray-800">${review.name}</h3>
                <p class="text-gray-600 mt-2 italic">"${review.text}"</p>
                <p class="text-gray-500 mt-2">📩 <a href="mailto:${review.email}" class="text-blue-500">${review.email}</a></p>
                <button class="delete-btn absolute top-2 right-2 bg-red-500 text-white px-3 py-1 rounded">X</button>
            `;
    
            let deleteBtn = reviewBox.querySelector(".delete-btn");
            deleteBtn.addEventListener("click", function () {
                let passcode = prompt("Enter admin passcode to request review deletion:");
    
                if (passcode !== OWNER_PASSCODE) {
                    alert("Incorrect passcode! You cannot request deletion.");
                    return;
                }
    
                sendDeletionRequest(review, index);  // No need to ask for passcode again
                });
    
    
            reviewCarousel.appendChild(reviewBox);
        });
    }
    
    function sendDeletionRequest(review, index) {
    emailjs.send("service_r2wla0l", "template_5c8c3an", {
        to_name: "Yogesh Gupta",  
        from_name: review.name,  
        message: `Request to delete review:\n\n"${review.text}"`,
        reviewer_email: review.email, 
        review_index: index, 
        reply_to: review.email
    }).then(
        function(response) {
            alert("Deletion request sent successfully.");
            console.log("Email sent:", response);
        },
        function(error) {
            alert("Failed to send deletion request. Try again.");
            console.error("Email error:", error);
        }
    );
    }
    
    
    
    
    reviewForm.addEventListener("submit", function (e) {
        e.preventDefault();
        let name = document.getElementById("name").value.trim();
        let email = document.getElementById("email").value.trim();
        let text = document.getElementById("review").value.trim();
    
        if (name && email && text) {
            let newReview = { name, email, text };
            reviews.push(newReview);
            localStorage.setItem("submittedReviews", JSON.stringify(reviews));
    
            reviewForm.reset();
            toggleReviewSection();
            displayReviews();
        }
    });
    
    displayReviews();
    
    // ✅ **NEW: Check if URL contains a deletion approval request**
    const params = new URLSearchParams(window.location.search);
    const approveIndex = params.get("approve_deletion");
    
    if (approveIndex !== null) {
        let passcode = prompt("Enter admin passcode to confirm deletion:");
    
        if (passcode === OWNER_PASSCODE) {
            let reviews = JSON.parse(localStorage.getItem("submittedReviews")) || [];
            
            if (approveIndex >= 0 && approveIndex < reviews.length) {
                reviews.splice(approveIndex, 1);  // Remove the review
                localStorage.setItem("submittedReviews", JSON.stringify(reviews));
                
                alert("Review deleted successfully.");
                location.href = "index.html";  // Redirect back to remove the URL parameter
            } else {
                alert("Invalid review index. Deletion failed.");
            }
        } else {
            alert("Incorrect passcode! Deletion denied.");
        }
    }
    });